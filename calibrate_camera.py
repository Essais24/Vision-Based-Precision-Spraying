import argparse
import glob
import os
import cv2
import numpy as np

DEFAULT_NPZ = "/home/archi/Desktop/BTP-I/camera_calibration.npz"
DEFAULT_IMG_FOLDER = "/home/archi/Desktop/BTP-I/calibration_images"


def get_charuco_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def load_intrinsics(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing calibration file: {npz_path}")
    with np.load(npz_path) as data:
        camera_matrix = data.get('K') if data.get('K') is not None else data.get('camera_matrix')
        dist_coeffs = data.get('dist') if data.get('dist') is not None else data.get('dist_coeffs')
        if camera_matrix is None or dist_coeffs is None:
            raise KeyError("Could not find keys in the .npz file.")
    return camera_matrix, dist_coeffs


def calibrate_intrinsics(image_folder, output_npz=DEFAULT_NPZ):
    """Calibrate camera intrinsics (K, dist) from a collection of ChArUco images."""
    board, detector = get_charuco_board()
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png")))
    if not image_paths:
        print(f"[ERROR] No images found inside: {image_folder}")
        return None, None

    print(f"Found {len(image_paths)} images in {image_folder}. Detecting ChArUco corners...")
    obj_points = []
    img_points = []
    img_size = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        if corners is not None and len(corners) >= 6:
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            if obj_pts is not None and len(obj_pts) >= 6:
                obj_points.append(obj_pts)
                img_points.append(img_pts)
                print(f"  ✓ {os.path.basename(path)}: {len(corners)} corners matched")
        else:
            print(f"  ✗ {os.path.basename(path)}: insufficient corners detected")

    if len(obj_points) < 4:
        print(f"[ERROR] Only {len(obj_points)} valid frames with ChArUco board found. Need at least 4 (15+ recommended).")
        return None, None

    print(f"\nRunning camera calibration across {len(obj_points)} frames ({img_size[0]}x{img_size[1]})...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    print("\n" + "=" * 60)
    print(f"INTRINSIC CALIBRATION RESULTS (RMS Error: {ret:.4f} px)")
    print("=" * 60)
    print(f"K (Camera Matrix):\n{np.array2string(K, separator=', ')}")
    print(f"\nDistortion Coefficients (dist):\n{np.array2string(dist.flatten(), separator=', ')}")

    # Save to npz and .npy files
    np.savez(output_npz, K=K, dist=dist, image_size=np.array(img_size), rms=ret)
    save_dir = os.path.join(os.path.dirname(output_npz), "camera_calibration")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "K.npy"), K)
    np.save(os.path.join(save_dir, "dist.npy"), dist)
    np.save(os.path.join(save_dir, "image_size.npy"), np.array(img_size))
    np.save(os.path.join(save_dir, "rms.npy"), ret)
    print(f"\n[SAVED] Saved intrinsics to {output_npz} and {save_dir}/")
    return K, dist


def process_images_for_extrinsics(image_folder, camera_matrix, dist_coeffs):
    """Estimate ground-plane extrinsics (R, t, height) from image(s) with board on ground."""
    board, detector = get_charuco_board()
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png")))
    if not image_paths:
        print(f"No JPG/PNG images found inside: {image_folder}")
        return

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_corners is not None and len(charuco_corners) > 4:
            board_corners = board.getChessboardCorners()
            object_points = board_corners[charuco_ids.flatten()].reshape(-1, 1, 3)
            valid, rvec, tvec = cv2.solvePnP(
                object_points,
                charuco_corners,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if valid:
                R_board_to_cam, _ = cv2.Rodrigues(rvec)
                R_cam_to_board = R_board_to_cam.T
                camera_position_world = -R_cam_to_board @ tvec

                print("\n" + "=" * 60)
                print(f"CAMERA HEIGHT MEASUREMENT: {os.path.basename(path)}")
                print("=" * 60)
                measured_height = abs(camera_position_world[2][0])
                print(f"Computed Camera Height: {measured_height:.4f} m ({measured_height * 100:.1f} cm)")
                print(f"Update CAMERA_HEIGHT_M = {measured_height:.4f} in detection_worldcoord.py")
                print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration utility for top-down weed spraying pipeline.")
    parser.add_argument("--images", default=DEFAULT_IMG_FOLDER, help=f"Folder of calibration images (default: {DEFAULT_IMG_FOLDER})")
    parser.add_argument("--intrinsics", action="store_true", default=True, help="Run intrinsic calibration (default: True)")
    parser.add_argument("--measure-height", action="store_true", help="Estimate camera height from a ground ChArUco image")
    parser.add_argument("--npz", default=DEFAULT_NPZ, help=f"Path to camera_calibration.npz (default: {DEFAULT_NPZ})")
    args = parser.parse_args()

    if args.measure_height:
        K_matrix, dist_vector = load_intrinsics(args.npz)
        process_images_for_extrinsics(args.images, K_matrix, dist_vector)
    else:
        calibrate_intrinsics(args.images, args.npz)


if __name__ == "__main__":
    main()
