import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from face_mask_detection.data import _IMAGE_SUFFIX
from face_mask_detection.dvc_utils import ensure_data_available


@dataclass
class ImageExample:
    # Represents a single image and its associated label, along with the classes present in that image.
    image_path: Path
    label_path: Path
    classes: Counter = field(default_factory=Counter)


@dataclass
class ImageGroup:
    # Represents a group of images that should be kept together in the same split (e.g. all images of the same person).
    group_id: str
    examples: list[ImageExample] = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)

    def image_count(self):
        return len(self.examples)


def split_data(cfg):
    dataset_dir = ensure_data_available(cfg)
    split_names = [
        str(cfg.data.train_split),
        str(cfg.data.val_split),
        str(cfg.data.test_split),
    ]
    ratios = _split_ratios(cfg, split_names)
    groups = _collect_groups(dataset_dir, split_names)

    # Find the best group-to-split assignments.
    assignments = _find_best_assignments(
        groups=groups,
        split_names=split_names,
        ratios=ratios,
        seed=int(cfg.seed),
        iterations=int(cfg.data.get("split_search_iterations", 20_000)),
    )
    # Do the actual split, write summary, and print results.
    _write_split_folders(dataset_dir, split_names, groups, assignments)
    summary = _split_summary(groups, assignments, split_names, ratios, int(cfg.seed))
    _write_summary(dataset_dir, summary)
    _print_summary(summary)
    return dataset_dir


def _split_ratios(cfg, split_names):
    ratios = {
        split_names[0]: float(cfg.data.get("split_train_ratio", 0.70)),
        split_names[1]: float(cfg.data.get("split_val_ratio", 0.15)),
        split_names[2]: float(cfg.data.get("split_test_ratio", 0.15)),
    }
    ratio_sum = sum(ratios.values())
    if ratio_sum != 1.0:
        raise ValueError(f"Split ratios must sum to 1.0, but got {ratio_sum}")
    return {split: ratio / ratio_sum for split, ratio in ratios.items()}


def _collect_groups(dataset_dir, split_names):
    # Collect all images and their classes.
    groups = {}
    for split in split_names:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing split images directory: {images_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing split labels directory: {labels_dir}")

        for image_path in sorted(images_dir.iterdir()):
            # Only consider files with supported image extensions
            if image_path.suffix.lower() != _IMAGE_SUFFIX:
                continue

            # Each image must have a corresponding label file with the same stem name and .txt extension
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

            # Group ID is determined by the image filename, e.g. "maksssksksss0" for "maksssksksss0.png"
            # Group all images with the same group ID together, and collect the classes present in those images.
            group_id = _group_id(image_path)
            group = groups.setdefault(group_id, ImageGroup(group_id=group_id))
            classes = _read_label_classes(label_path)
            example = ImageExample(
                image_path=image_path,
                label_path=label_path,
                classes=classes,
            )
            group.examples.append(example)
            group.classes.update(classes)

    if not groups:
        raise ValueError(f"No image groups found under {dataset_dir}")
    return groups


def _group_id(image_path):
    # Regex to extract group id
    match = re.search(r"(maksssksksss\d+)$", image_path.stem)
    if match:
        return match.group(1)
    return image_path.stem


def _read_label_classes(label_path):
    # Reads the label file and counts the occurrences of each class in that file.
    classes = Counter()
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO label at {label_path}:{line_number}: {line}"
            )
        classes[int(float(parts[0]))] += 1
    return classes


def _find_best_assignments(groups, split_names, ratios, seed, iterations):
    # We want to assign each group into exactly one split, while trying to match the target ratios for both total images and class distributions.
    group_ids = sorted(groups)
    target_sizes = _target_group_counts(len(group_ids), split_names, ratios)

    best_score = None
    best_assignments = None
    for offset in range(iterations):
        random_generator = random.Random(seed + offset)
        shuffled_group_ids = group_ids[:]
        random_generator.shuffle(shuffled_group_ids)
        assignments = _assign_by_target_sizes(
            shuffled_group_ids, split_names, target_sizes
        )
        score = _assignment_score(groups, assignments, split_names, ratios)
        if best_score is None or score < best_score:
            best_score = score
            best_assignments = assignments

    return best_assignments


def _target_group_counts(total_groups, split_names, ratios):
    # Calculate the target number of groups for each split based on the ratios, and handle rounding to ensure the total matches.
    raw_counts = {split: total_groups * ratios[split] for split in split_names}
    counts = {split: int(raw_counts[split]) for split in split_names}
    remaining = total_groups - sum(counts.values())

    ranked_splits = sorted(
        split_names,
        key=lambda split: (raw_counts[split] - counts[split], split),
        reverse=True,
    )
    for split in ranked_splits[:remaining]:
        counts[split] += 1
    return counts


def _assign_by_target_sizes(group_ids, split_names, target_sizes):
    # Assign groups to splits based on the target sizes. This assumes group_ids is already shuffled.
    assignments = {}
    start = 0
    for split in split_names:
        end = start + target_sizes[split]
        assignments[split] = group_ids[start:end]
        start = end
    return assignments


def _assignment_score(groups, assignments, split_names, ratios):
    # Assignment score based on how well it matches the target ratios
    # Uses relative square error for both total images and class distributions.
    total_images = sum(group.image_count() for group in groups.values())
    total_classes = Counter()
    for group in groups.values():
        total_classes.update(group.classes)

    score = 0.0
    for split in split_names:
        split_groups = [groups[group_id] for group_id in assignments[split]]
        split_images = sum(group.image_count() for group in split_groups)
        image_target = total_images * ratios[split]
        score += _relative_square_error(split_images, image_target)

        split_classes = Counter()
        for group in split_groups:
            split_classes.update(group.classes)
        for class_id, total_count in total_classes.items():
            class_target = total_count * ratios[split]
            score += _relative_square_error(split_classes[class_id], class_target)
    return score


def _relative_square_error(actual, target):
    return ((actual - target) / max(1.0, target)) ** 2


def _write_split_folders(dataset_dir, split_names, groups, assignments):
    # Write the assigned splits to temporary folders first, then move them to the final location.
    # This ensures that we don't end up with a partially split dataset if something goes wrong during copying.
    with tempfile.TemporaryDirectory(prefix="split-", dir=dataset_dir) as temp_dir:
        temp_path = Path(temp_dir)
        for split in split_names:
            images_dir = temp_path / split / "images"
            labels_dir = temp_path / split / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            for group_id in sorted(assignments[split]):
                for example in sorted(
                    groups[group_id].examples,
                    key=lambda item: item.image_path.name,
                ):
                    shutil.copy2(
                        example.image_path, images_dir / example.image_path.name
                    )
                    shutil.copy2(
                        example.label_path, labels_dir / example.label_path.name
                    )

        for split in split_names:
            split_dir = dataset_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            shutil.move(str(temp_path / split), split_dir)


def _split_summary(groups, assignments, split_names, ratios, seed):
    # Generate a summary of the split, including total groups, total images, leakage count, and per-split statistics.
    total_images = sum(group.image_count() for group in groups.values())
    total_groups = len(groups)
    leakage_count = _leakage_count(assignments)

    summary = {
        "seed": seed,
        "ratios": {split: ratios[split] for split in split_names},
        "total_groups": total_groups,
        "total_images": total_images,
        "leakage_groups": leakage_count,
        "splits": {},
    }
    for split in split_names:
        split_groups = [groups[group_id] for group_id in assignments[split]]
        object_counts = Counter()
        for group in split_groups:
            object_counts.update(group.classes)
        image_count = sum(group.image_count() for group in split_groups)
        summary["splits"][split] = {
            "groups": len(split_groups),
            "images": image_count,
            "image_percentage": round(image_count / total_images, 4),
            "objects": {int(key): int(value) for key, value in object_counts.items()},
        }
    return summary


def _leakage_count(assignments):
    # Count how many groups are assigned to more than one split, which indicates leakage.
    group_to_splits = defaultdict(set)
    for split, group_ids in assignments.items():
        for group_id in group_ids:
            group_to_splits[group_id].add(split)
    return sum(1 for split_set in group_to_splits.values() if len(split_set) > 1)


def _write_summary(dataset_dir, summary):
    summary_path = dataset_dir / "split_summary.yaml"
    OmegaConf.save(OmegaConf.create(summary), summary_path)


def _print_summary(summary):
    print("Stratified group split complete")
    print(f"Total groups: {summary['total_groups']}")
    print(f"Total images: {summary['total_images']}")
    print(f"Leakage groups: {summary['leakage_groups']}")
    for split, split_summary in summary["splits"].items():
        percentage = split_summary["image_percentage"] * 100
        print(
            f"- {split}: {split_summary['groups']} groups, "
            f"{split_summary['images']} images ({percentage:.1f}%)"
        )
