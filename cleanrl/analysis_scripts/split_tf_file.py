import os
import tensorflow as tf


tf.compat.v1.disable_eager_execution()


def split_event_file(input_event_file: str, output_directory: str):
    """
    Reads a combined TensorBoard event file and splits its events into separate event files.
    Each file is written directly into the specified output directory (without creating subfolders)
    and is prefixed with the run identifier extracted from the event tags.

    Parameters:
      input_event_file: str
          Path to the input event file that contains logs from multiple runs.
      output_directory: str
          Directory where the split event files will be saved.
          This directory must already exist.
    """
    # A dictionary mapping run_id -> tf FileWriter.
    writers = {}

    # Iterate over the combined event file.
    for event in tf.compat.v1.train.summary_iterator(input_event_file):
        run_id = None
        # Check if the event has a summary field.
        if event.HasField("summary"):
            for value in event.summary.value:
                # Look for a tag of the form "run_prefix/..."
                if "/" in value.tag:
                    run_id = value.tag.split("/")[0]
                    break
        # If no run_id is found, group into a default run.
        if run_id is None:
            run_id = "default"

        # If this run_id hasn't been seen yet, create a new writer.
        if run_id not in writers:
            base_name = os.path.basename(input_event_file)
            # File will be named like "seed_123456_your_combined_event_file.tfevents.XXXX"
            new_event_file = os.path.join(output_directory, f"{run_id}_{base_name}")
            writers[run_id] = tf.compat.v1.summary.FileWriter(new_event_file)
            print(f"Created writer for run '{run_id}' in file: {new_event_file}")

        # Write the current event to the appropriate event file.
        writers[run_id].add_event(event)

    # Close all writers when done.
    for run_id, writer in writers.items():
        writer.close()
        print(f"Closed writer for run '{run_id}'")


if __name__ == "__main__":
    # Path to the combined tensorboard event file (old file with 5 runs)
    combined_event_file = "../runs/MinAtar/Asterix-v1__sac_min_atar_max_alpha_multi_run/events.out.tfevents.1744396844.DESKTOP-3KSSRPS.25828.0"  # <-- update this path
    # Output directory where separate run folders will be created.
    output_dir = "../runs/MinAtar/Asterix-v1__sac_min_atar_max_alpha_multi_run"

    split_event_file(combined_event_file, output_dir)
    print("Splitting complete!")
