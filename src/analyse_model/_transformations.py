def images_minmax_inv(images, normalisation_df):
    images = (
        images
        * (
            normalisation_df.loc["images"].maximum
            - normalisation_df.loc["images"].minimum
        )
        + normalisation_df.loc["images"].minimum
    )
    return images


def images_zscore_inv(images, normalisation_df):
    images = (
        images * (normalisation_df.loc["images"].stds)
        + normalisation_df.loc["images"].means
    )
    return images


def images_zscore__minmax_inv(images, normalisation_df):
    images = images_minmax_inv(images, normalisation_df)
    images = images_zscore_inv(images, normalisation_df)

    return images


def metadata_minmax_inv(metadata, normalisation_df):
    metadata = (
        metadata
        * (
            normalisation_df.loc["metadata"].maximum
            - normalisation_df.loc["metadata"].minimum
        )
        + normalisation_df.loc["metadata"].minimum
    )
    return metadata


def metadata_zscore_inv(metadata, normalisation_df):
    metadata = (
        metadata * (normalisation_df.loc["metadata"].stds)
        + normalisation_df.loc["metadata"].means
    )
    return metadata


def metadata_zscore__minmax_inv(metadata, normalisation_df):
    metadata = metadata_minmax_inv(metadata, normalisation_df)
    metadata = metadata_zscore_inv(metadata, normalisation_df)

    return metadata


def targets_minmax_inv(targets, normalisation_df):
    targets = (
        targets
        * (
            normalisation_df.loc["targets"].maximum
            - normalisation_df.loc["targets"].minimum
        )
        + normalisation_df.loc["targets"].minimum
    )
    return targets


def targets_zscore_inv(targets, normalisation_df):
    targets = (
        targets * (normalisation_df.loc["targets"].stds)
        + normalisation_df.loc["targets"].means
    )
    return targets


def targets_zscore__minmax_inv(targets, normalisation_df):
    targets = targets_minmax_inv(targets, normalisation_df)
    targets = targets_zscore_inv(targets, normalisation_df)

    return targets


def all_inverse_values(images,metadata,targets,normalisation_df):
    images_inverse = images_zscore__minmax_inv(
        images,normalisation_df
        )
    metadata_inverse = metadata_zscore__minmax_inv(
            metadata, normalisation_df
        )
    targets_inverse = targets_zscore__minmax_inv(
            targets, normalisation_df
        )
    true_lengths = targets_inverse[:, 0].astype(int)
    true_widths = targets_inverse[:, 1].astype(int)
    true_ratios = targets_inverse[:, 2].astype(int)

    return images_inverse,metadata_inverse,targets_inverse,true_lengths,true_widths,true_ratios
