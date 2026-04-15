#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn as nn
from random import randint
import random
from utils.loss_utils import l1_loss, ssim, cos_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vq_utils import load_2d_language_feature, ResidualVectorQuantizationWithClustering
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
    
    # Initialize language feature codebooks
    if opt.include_feature and first_iter == 0:
        device = torch.device("cuda")
        features = load_2d_language_feature(dataset.lf_path, device)
        rvq = ResidualVectorQuantizationWithClustering(opt.vq_layer_num, opt.codebook_size, features.shape[1], device).to(device)
        rvq.fit_quantizers(features)
        codebooks = torch.stack(rvq.quantizers, dim=0).to(device)
        with torch.no_grad():
            # Global branch init (legacy alias points to global too)
            if getattr(gaussians, "_global_language_feature_codebooks", None) is not None:
                gaussians._global_language_feature_codebooks.data.copy_(codebooks)
                gaussians._language_feature_codebooks = gaussians._global_language_feature_codebooks
            else:
                gaussians._language_feature_codebooks.data.copy_(codebooks)

            # Local branch init
            if getattr(gaussians, "_local_language_feature_codebooks", None) is not None:
                init_mode = str(getattr(opt, "local_codebook_init_mode", "copy_global"))
                if init_mode == "random":
                    pass
                else:
                    # default: copy global codebooks to every region for stable start
                    R = gaussians._local_language_feature_codebooks.shape[0]
                    gaussians._local_language_feature_codebooks.data.copy_(codebooks.unsqueeze(0).repeat(R, 1, 1, 1))

    # Precompute local region masks once (feature training freezes xyz/opacity/etc.)
    # Used only for sampling which regions to reconstruct (no longer used for per-region render).
    region_masks = None
    non_empty_region_ids = None
    if opt.include_feature:
        num_regions = int(getattr(opt, "num_local_regions", 1))
        if num_regions > 1 and getattr(gaussians, "_local_region_ids", None) is not None:
            region_masks = [gaussians.get_local_region_mask(rid) for rid in range(num_regions)]
            non_empty_region_ids = [rid for rid, m in enumerate(region_masks) if bool(m.any().item())]

        # One-time performance hint for default settings.
        if num_regions > 1 and int(getattr(opt, "local_region_sample_num", -1)) < 0 and int(getattr(opt, "local_render_interval", 1)) <= 1:
            print(
                f"[perf-hint] num_local_regions={num_regions} 且每次迭代都计算全部 local 分支，"
                f"训练耗时通常会接近 (1+R) 倍。可尝试："
                f"--local_region_sample_num 2 或 --local_render_interval 2 以显著降耗时。"
            )

        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss_record = []
    iter_record = []
    smooth_loss = None
    for iteration in range(first_iter, opt.iterations + 1):        
        do_time_breakdown = bool(getattr(args, "time_breakdown", False))
        time_every = int(getattr(args, "time_breakdown_every", 50))
        time_first = int(getattr(args, "time_breakdown_first", 0))
        profile_this_iter = do_time_breakdown and (iteration <= max(time_first, 0) or (time_every > 0 and iteration % time_every == 0))
        if profile_this_iter:
            torch.cuda.synchronize()
            t_iter0 = time.perf_counter()
            # Initialize timestamps to avoid UnboundLocalError in branches.
            t_render = t_iter0
            t_gt = t_iter0
            t_global = t_iter0
            t_local = t_iter0
            t_bwd = t_iter0
            t_opt = t_iter0

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, opt, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # Keep compatibility with legacy CLI flag `--topk`.
        # If user only sets --topk, propagate it to both global/local topk.
        if hasattr(args, "topk"):
            if hasattr(args, "global_topk") and int(args.global_topk) == int(getattr(opt, "global_topk", 1)):
                opt.global_topk = int(args.topk)
            else:
                opt.global_topk = int(getattr(args, "global_topk", getattr(opt, "global_topk", 1)))

            if hasattr(args, "local_topk") and int(args.local_topk) == int(getattr(opt, "local_topk", 1)):
                opt.local_topk = int(args.topk)
            else:
                opt.local_topk = int(getattr(args, "local_topk", getattr(opt, "local_topk", 1)))

        # Decide local training strategy before rendering.
        num_regions = int(getattr(opt, "num_local_regions", 1))
        local_interval = int(getattr(opt, "local_render_interval", 1))
        sample_num = int(getattr(opt, "local_region_sample_num", -1))
        do_local = (local_interval <= 1) or (iteration % local_interval == 0)
        has_local_params = (getattr(gaussians, "_local_language_feature_codebooks", None) is not None) and (getattr(gaussians, "_local_language_feature_logits", None) is not None)
        has_regions = (region_masks is not None) and (non_empty_region_ids is not None) and (len(non_empty_region_ids) > 0)
        enable_local = opt.include_feature and do_local and num_regions > 1 and has_local_params and has_regions and (sample_num != 0)

        # Render policy:
        # - If local is disabled this iter -> render global (64ch) only.
        # - If local is enabled and sample_num is small -> render global once + K local renders (avoid packed 576ch backward).
        # - Otherwise -> render packed once (fastest when needing all regions).
        use_sampled_multi_render = False
        chosen_rids = []
        if enable_local:
            if sample_num > 0 and sample_num < len(non_empty_region_ids):
                use_sampled_multi_render = True
                chosen_rids = random.sample(non_empty_region_ids, sample_num)
            elif sample_num > 0 and sample_num >= len(non_empty_region_ids):
                chosen_rids = non_empty_region_ids
            elif sample_num < 0:
                chosen_rids = non_empty_region_ids

        if (not enable_local) or use_sampled_multi_render:
            # Main render: global-only (base channels).
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt, language_branch="global")
        else:
            # Full local: packed once.
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt, language_branch="packed")

        image, weight_map, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["language_feature_weight_map"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        if profile_this_iter:
            torch.cuda.synchronize()
            t_render = time.perf_counter()
        
        # Loss
        if opt.include_feature:
            # gt_language_feature [512 H W]
            gt_language_feature, language_feature_mask = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=dataset.feature_level)

            if profile_this_iter:
                torch.cuda.synchronize()
                t_gt = time.perf_counter()

            # In this paper, we select layer_num = 1
            layer_num, _, _ = gaussians.get_language_feature_codebooks.shape
            layer_idx = min(int(iteration / 10000 * layer_num), layer_num - 1)

            # Global feature reconstruction (auto-slices packed map)
            global_feature = gaussians.compute_global_layer_feature_map(weight_map, layer_idx)

            if profile_this_iter:
                torch.cuda.synchronize()
                t_global = time.perf_counter()

            # Local feature reconstruction
            alpha = float(getattr(opt, "global_local_alpha", 0.5))
            alpha_eff = alpha
            local_feature = torch.zeros_like(global_feature)

            if not enable_local:
                alpha_eff = 1.0
                if profile_this_iter:
                    t_local = t_global
            else:
                if use_sampled_multi_render:
                    # Sampled local: K local renders with base channels (64) to avoid packed backward overhead.
                    for rid in chosen_rids:
                        local_pkg = render(
                            viewpoint_cam,
                            gaussians,
                            pipe,
                            background,
                            opt,
                            language_branch="local",
                            gaussian_mask=region_masks[int(rid)],
                        )
                        local_w = local_pkg["language_feature_weight_map"]
                        local_feature = local_feature + gaussians.compute_local_layer_feature_map(local_w, layer_idx, int(rid))

                    if len(chosen_rids) > 0 and len(chosen_rids) < len(non_empty_region_ids):
                        local_feature = local_feature * (float(len(non_empty_region_ids)) / float(len(chosen_rids)))
                else:
                    # Packed local: decode from packed weight map.
                    layer_num, codebook_size, _ = gaussians.get_language_feature_codebooks.shape
                    base_C = int(layer_num * codebook_size)
                    expected_C = base_C * (1 + int(num_regions))
                    if int(weight_map.shape[0]) < expected_C:
                        alpha_eff = 1.0
                        if not hasattr(training, "_warned_packed_mismatch"):
                            print(
                                f"[warn] language_feature_weight_map 通道数={int(weight_map.shape[0])}，"
                                f"不足以支持 packed local-global（期望 >= {expected_C}）。已退化为 global-only。"
                            )
                            training._warned_packed_mismatch = True
                    else:
                        for rid in chosen_rids:
                            start = base_C + int(rid) * base_C
                            end = start + base_C
                            local_slice = weight_map[start:end]
                            local_feature = local_feature + gaussians.compute_local_layer_feature_map(local_slice, layer_idx, int(rid))

                        if len(chosen_rids) > 0 and len(chosen_rids) < len(non_empty_region_ids):
                            local_feature = local_feature * (float(len(non_empty_region_ids)) / float(len(chosen_rids)))

            if profile_this_iter:
                torch.cuda.synchronize()
                t_local = time.perf_counter()

            language_feature = alpha_eff * global_feature + (1.0 - alpha_eff) * local_feature
            if args.normalize:
                language_feature = language_feature / (language_feature.norm(dim=0, keepdim=True) + 1e-10)
            loss = 0
            if args.cos_loss:
                cosloss = cos_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)
                loss += cosloss
            if args.l1_loss:
                Ll1 = l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)   
                loss += Ll1

        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            if profile_this_iter:
                torch.cuda.synchronize()
                t_gt = t_render
                t_global = t_render
                t_local = time.perf_counter()
        loss.backward()
        iter_end.record()

        if profile_this_iter:
            torch.cuda.synchronize()
            t_bwd = time.perf_counter()
        
        iter_record.append(iteration)
        if smooth_loss is None:
            smooth_loss = loss.item()
        else:
            smooth_loss = smooth_loss * 0.99 + loss.item() * 0.01
        loss_record.append(smooth_loss)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, opt))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if not opt.include_feature:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if (iteration < opt.iterations) and (iteration % args.accum_iter == 0):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if profile_this_iter:
                torch.cuda.synchronize()
                t_opt = time.perf_counter()
                wshape = tuple(weight_map.shape) if (opt.include_feature and isinstance(weight_map, torch.Tensor)) else None
                msg = (
                    f"[time] iter={iteration} "
                    f"render={(t_render - t_iter0):.3f}s "
                    f"gt={(t_gt - t_render):.3f}s "
                    f"global={(t_global - t_gt):.3f}s "
                    f"local={(t_local - t_global):.3f}s "
                    f"bwd={(t_bwd - t_local):.3f}s "
                    f"opt={(t_opt - t_bwd):.3f}s "
                    f"total={(t_opt - t_iter0):.3f}s "
                    f"weight_map_shape={wshape}"
                )
                print(msg)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if iteration == 10000:
                    return
            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55557)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--cos_loss', action='store_true', default=False)
    parser.add_argument('--l1_loss', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--topk', type=int, default=1)
    # Profiling helpers
    parser.add_argument('--time_breakdown', action='store_true', default=False)
    parser.add_argument('--time_breakdown_every', type=int, default=50, help='Print breakdown every N iters (<=0 disables periodic printing)')
    parser.add_argument('--time_breakdown_first', type=int, default=0, help='Also print breakdown for first N iters (0 disables)')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path + f"_{str(args.feature_level)}"
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    # All done
    print("\nTraining complete.")
