from pathlib import Path
import json
import math

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESIZE_TO = 28
CONNECTIVITY = 8
MIN_AREA = 3
MAX_GAP = 3.0


def read_image(path: Path, resize_to: int = 28) -> np.ndarray:
    img_bgr = cv2.imread(str(path))

    if img_bgr is None:
        raise ValueError(f"Gagal membaca gambar: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if resize_to is not None:
        img_rgb = cv2.resize(
            img_rgb,
            (resize_to, resize_to),
            interpolation=cv2.INTER_AREA
        )

    return img_rgb


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    min_val = float(x.min())
    max_val = float(x.max())

    if max_val - min_val < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return (x - min_val) / (max_val - min_val)


def otsu_threshold_float(score: np.ndarray) -> float:
    score_u8 = np.clip(score * 255, 0, 255).astype(np.uint8)

    threshold_value, _ = cv2.threshold(
        score_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return threshold_value / 255.0

def bacilli_color_score(img_rgb: np.ndarray) -> np.ndarray:
    img_float = img_rgb.astype(np.float32) / 255.0

    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]

    red_excess = np.clip(R - 0.5 * (G + B), 0, 1)
    purple_excess = np.clip((R + B) / 2.0 - G, 0, 1)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0

    darkness = 1.0 - V

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    A = lab[:, :, 1].astype(np.float32)

    a_red = np.clip((A - 128.0) / 127.0, 0, 1)

    red_hue = ((H <= 12) | (H >= 165)).astype(np.float32)
    purple_hue = ((H >= 125) & (H <= 165)).astype(np.float32)
    hue_score = np.maximum(red_hue, purple_hue) * S

    score = (
        0.35 * normalize_01(a_red)
        + 0.20 * normalize_01(S)
        + 0.20 * normalize_01(red_excess)
        + 0.15 * normalize_01(purple_excess)
        + 0.05 * normalize_01(darkness)
        + 0.05 * normalize_01(hue_score)
    )

    return normalize_01(score)


def proposed_mask_from_score(
    img_rgb: np.ndarray,
    min_area: int = 3,
    close_iter: int = 1,
    open_iter: int = 0
) -> tuple[np.ndarray, np.ndarray]:

    score = bacilli_color_score(img_rgb)

    threshold_otsu = otsu_threshold_float(score)
    threshold_percentile = float(np.percentile(score, 80))

    threshold_final = max(threshold_otsu, threshold_percentile * 0.85)

    mask = (score >= threshold_final).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    if close_iter > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iter
        )

    if open_iter > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iter
        )

    mask = filter_small_components(mask, min_area=min_area)

    return mask, score


def channel_threshold(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.uint8)

    _, mask = cv2.threshold(
        channel,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return (mask > 0).astype(np.uint8)


def boolean_masks(img_rgb: np.ndarray, min_area: int = 3) -> dict:

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    mR = channel_threshold(R)
    mG = channel_threshold(G)
    mB = channel_threshold(B)

    mask_or = ((mR | mG | mB) > 0).astype(np.uint8)
    mask_and = ((mR & mG & mB) > 0).astype(np.uint8)
    mask_xor = ((mR ^ mG ^ mB) > 0).astype(np.uint8)
    mask_xnor = (1 - mask_xor).astype(np.uint8)

    return {
        "RGB_OR": filter_small_components(mask_or, min_area=min_area),
        "RGB_AND": filter_small_components(mask_and, min_area=min_area),
        "RGB_XOR": filter_small_components(mask_xor, min_area=min_area),
        "RGB_XNOR": filter_small_components(mask_xnor, min_area=min_area),
    }



def filter_small_components(mask: np.ndarray, min_area: int = 3) -> np.ndarray:

    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    output = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            output[labels == label_id] = 1

    return output

def get_neighbors(
    row: int,
    col: int,
    height: int,
    width: int,
    connectivity: int = 8
):

    if connectivity == 4:
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]
    else:
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1)
        ]

    for dr, dc in directions:
        nr = row + dr
        nc = col + dc

        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def component_orientation(coords: np.ndarray) -> float:

    if len(coords) < 2:
        return 0.0

    xy = np.column_stack([coords[:, 1], coords[:, 0]]).astype(np.float32)
    xy = xy - xy.mean(axis=0, keepdims=True)

    cov = np.cov(xy.T)

    try:
        eigvals, eigvecs = np.linalg.eig(cov)
        main_vec = eigvecs[:, np.argmax(eigvals)]
        angle = math.atan2(main_vec[1], main_vec[0])
    except Exception:
        angle = 0.0

    return angle


def angle_diff(angle_a: float, angle_b: float) -> float:

    diff = abs(angle_a - angle_b) % math.pi
    return min(diff, math.pi - diff)


def find_endpoints(mask: np.ndarray, connectivity: int = 8) -> list:

    height, width = mask.shape
    endpoints = []

    object_pixels = np.argwhere(mask > 0)

    for row, col in object_pixels:
        degree = 0

        for nr, nc in get_neighbors(
            int(row),
            int(col),
            height,
            width,
            connectivity
        ):
            if mask[nr, nc] > 0:
                degree += 1

        if degree <= 1:
            endpoints.append((int(row), int(col)))

    return endpoints


def line_score(
    score: np.ndarray,
    point_1: tuple,
    point_2: tuple
) -> float:
    r1, c1 = point_1
    r2, c2 = point_2

    temp = np.zeros(score.shape, dtype=np.uint8)

    cv2.line(
        temp,
        (c1, r1),
        (c2, r2),
        1,
        thickness=1
    )

    values = score[temp > 0]

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def endpoint_bridging(
    mask: np.ndarray,
    score: np.ndarray,
    max_gap: float = 3.0,
    max_angle_deg: float = 35.0,
    min_line_score_ratio: float = 0.45
) -> np.ndarray:

    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 2:
        return mask

    component_angle = {}

    for label_id in range(1, num_labels):
        coords = np.argwhere(labels == label_id)
        component_angle[label_id] = component_orientation(coords)

    endpoints = []

    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(np.uint8)
        component_endpoints = find_endpoints(component_mask, connectivity=8)

        for endpoint in component_endpoints:
            endpoints.append((endpoint[0], endpoint[1], label_id))

    output = mask.copy()
    score_gate = float(np.percentile(score, 70)) * min_line_score_ratio

    for i in range(len(endpoints)):
        r1, c1, label_1 = endpoints[i]

        for j in range(i + 1, len(endpoints)):
            r2, c2, label_2 = endpoints[j]

            if label_1 == label_2:
                continue

            distance = math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

            if distance > max_gap:
                continue

            angle_difference = math.degrees(
                angle_diff(
                    component_angle[label_1],
                    component_angle[label_2]
                )
            )

            if angle_difference > max_angle_deg:
                continue

            avg_line_score = line_score(
                score,
                (r1, c1),
                (r2, c2)
            )

            if avg_line_score < score_gate:
                continue

            cv2.line(
                output,
                (c1, r1),
                (c2, r2),
                1,
                thickness=1
            )

    return output.astype(np.uint8)

def mask_to_graph(mask: np.ndarray, connectivity: int = 8) -> dict:

    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape

    coords = np.argwhere(mask > 0)

    node_id = {}

    for idx, (row, col) in enumerate(coords):
        node_id[(int(row), int(col))] = idx

    num_labels, labels = cv2.connectedComponents(
        mask,
        connectivity=connectivity
    )

    nodes = []

    for idx, (row, col) in enumerate(coords):
        row = int(row)
        col = int(col)

        nodes.append({
            "id": int(idx),
            "row": row,
            "col": col,
            "component": int(labels[row, col])
        })

    edges = []
    adjacency_list = {}

    for i in range(len(nodes)):
        adjacency_list[str(i)] = []

    if connectivity == 4:
        directions = [
            (1, 0),
            (0, 1)
        ]
    else:
        directions = [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1)
        ]

    for row, col in node_id.keys():
        u = node_id[(row, col)]

        for dr, dc in directions:
            nr = row + dr
            nc = col + dc

            if (nr, nc) in node_id:
                v = node_id[(nr, nc)]

                edges.append([int(u), int(v)])
                adjacency_list[str(u)].append(int(v))
                adjacency_list[str(v)].append(int(u))

    components = []

    for comp in range(1, num_labels):
        comp_pixels = np.argwhere(labels == comp)

        if len(comp_pixels) == 0:
            continue

        rows = comp_pixels[:, 0]
        cols = comp_pixels[:, 1]

        components.append({
            "component": int(comp),
            "node_count": int(len(comp_pixels)),
            "bbox": {
                "row_min": int(rows.min()),
                "row_max": int(rows.max()),
                "col_min": int(cols.min()),
                "col_max": int(cols.max())
            }
        })

    graph = {
        "height": int(height),
        "width": int(width),
        "connectivity": int(connectivity),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "component_count": int(num_labels - 1),
        "nodes": nodes,
        "edges": edges,
        "adjacency_list": adjacency_list,
        "components": components
    }

    return graph


def save_adjacency_matrix(
    graph: dict,
    output_path: Path,
    max_nodes_matrix: int = 1200
) -> bool:

    node_count = graph["node_count"]

    if node_count > max_nodes_matrix:
        return False

    adjacency_matrix = np.zeros(
        (node_count, node_count),
        dtype=np.uint8
    )

    for u, v in graph["edges"]:
        adjacency_matrix[u, v] = 1
        adjacency_matrix[v, u] = 1

    np.save(str(output_path), adjacency_matrix)

    return True

def save_mask_png(mask: np.ndarray, output_path: Path):
    cv2.imwrite(
        str(output_path),
        (mask * 255).astype(np.uint8)
    )


def save_visualization(
    img_rgb: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    graph: dict,
    output_path: Path
):

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input patch")
    axes[0].axis("off")

    axes[1].imshow(score, cmap="gray")
    axes[1].set_title("Bacilli score")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Mask akhir")
    axes[2].axis("off")

    axes[3].imshow(img_rgb)
    axes[3].set_title("Graph annotation")
    axes[3].axis("off")

    node_lookup = {}

    for node in graph["nodes"]:
        node_lookup[node["id"]] = (node["row"], node["col"])

    for u, v in graph["edges"]:
        r1, c1 = node_lookup[u]
        r2, c2 = node_lookup[v]

        axes[3].plot(
            [c1, c2],
            [r1, r2],
            linewidth=0.5
        )

    if len(graph["nodes"]) > 0:
        rows = [node["row"] for node in graph["nodes"]]
        cols = [node["col"] for node in graph["nodes"]]

        axes[3].scatter(cols, rows, s=4)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


def process_image(
    image_path: Path,
    output_dir: Path,
    resize_to: int = 28,
    connectivity: int = 8,
    min_area: int = 3,
    max_gap: float = 3.0
) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = image_path.stem

    img_rgb = read_image(
        image_path,
        resize_to=resize_to
    )

    boolean_dir = output_dir / "boolean_baseline"
    mask_dir = output_dir / "mask"
    graph_dir = output_dir / "graph_json"
    adjacency_dir = output_dir / "adjacency_matrix"
    visualization_dir = output_dir / "visualization"

    boolean_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    graph_dir.mkdir(exist_ok=True)
    adjacency_dir.mkdir(exist_ok=True)
    visualization_dir.mkdir(exist_ok=True)
    
    baseline_masks = boolean_masks(
        img_rgb,
        min_area=min_area
    )

    for method_name, baseline_mask in baseline_masks.items():
        save_mask_png(
            baseline_mask,
            boolean_dir / f"{image_name}_{method_name}.png"
        )

    mask_initial, score = proposed_mask_from_score(
        img_rgb,
        min_area=min_area
    )

    mask_bridged = endpoint_bridging(
        mask_initial,
        score,
        max_gap=max_gap
    )

    mask_final = filter_small_components(
        mask_bridged,
        min_area=min_area
    )

    graph = mask_to_graph(
        mask_final,
        connectivity=connectivity
    )

    graph["image_name"] = image_path.name
    graph["method"] = "Bacilli-Aware Graph Annotation"
    graph["min_area"] = int(min_area)
    graph["max_gap"] = float(max_gap)

    save_mask_png(
        mask_final,
        mask_dir / f"{image_name}_mask.png"
    )

    with open(
        graph_dir / f"{image_name}_graph.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(graph, file, indent=2)

    matrix_saved = save_adjacency_matrix(
        graph,
        adjacency_dir / f"{image_name}_adjacency.npy"
    )

    if not matrix_saved:
        edge_array = np.array(graph["edges"])

        np.savetxt(
            adjacency_dir / f"{image_name}_edge_list.csv",
            edge_array,
            fmt="%d",
            delimiter=","
        )

    save_visualization(
        img_rgb,
        score,
        mask_final,
        graph,
        visualization_dir / f"{image_name}_visualization.png"
    )

    result = {
        "image": image_path.name,
        "node_count": graph["node_count"],
        "edge_count": graph["edge_count"],
        "component_count": graph["component_count"]
    }

    return result

def process_folder(
    input_dir: Path,
    output_dir: Path,
    resize_to: int = 28,
    connectivity: int = 8,
    min_area: int = 3,
    max_gap: float = 3.0
):
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]

    image_paths = []

    for ext in extensions:
        image_paths.extend(input_dir.glob(ext))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"Tidak ada gambar jpg/png/bmp di folder: {input_dir}"
        )

    summary = []

    print("Graph Annotation Bacilli")
    print(f"Input folder  : {input_dir}")
    print(f"Output folder : {output_dir}")
    print(f"Jumlah gambar : {len(image_paths)}")

    for image_path in image_paths:
        print(f"Memproses: {image_path.name}")

        try:
            result = process_image(
                image_path=image_path,
                output_dir=output_dir,
                resize_to=resize_to,
                connectivity=connectivity,
                min_area=min_area,
                max_gap=max_gap
            )

            summary.append(result)

            print(
                f"  Node: {result['node_count']}, "
                f"Edge: {result['edge_count']}, "
                f"Komponen: {result['component_count']}"
            )

        except Exception as error:
            print(f"  Gagal memproses {image_path.name}: {error}")

    summary_path = output_dir / "summary.csv"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("image,node_count,edge_count,component_count\n")

        for row in summary:
            file.write(
                f"{row['image']},"
                f"{row['node_count']},"
                f"{row['edge_count']},"
                f"{row['component_count']}\n"
            )

    print("SELESAI")
    print(f"Output tersimpan di: {output_dir}")
    print(f"Summary tersimpan di: {summary_path}")

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Folder input tidak ditemukan: {INPUT_DIR}"
        )

    process_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        resize_to=RESIZE_TO,
        connectivity=CONNECTIVITY,
        min_area=MIN_AREA,
        max_gap=MAX_GAP
    )


if __name__ == "__main__":
    main()
