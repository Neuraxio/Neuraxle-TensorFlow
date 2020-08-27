import tensorflow as tf
import datetime
import json

for epoch in range(EPOCH):
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_values[epoch], step=epoch)


    with validation_summary_writer.as_default():
        tf.summary.scalar('loss', validation_values[epoch], step=epoch)

def test_repository_json_should_notify_when_trial_json_files_change(tmpdir):
    observer = Observer()
    repo_for_subscribing_to_trial_json_file_changes = HyperparamsJSONRepository(
        cache_folder=os.path.join(tmpdir, 'cache'),
        best_retrained_model_folder=os.path.join(tmpdir, 'best'),
    )
    # TODO: make HyperparamsRepository an Observable by inherting from your observable class: neuraxle/metaopt/auto_ml.py
    # TODO: make hyperparamsJsonrepository notify observers when trial json files change
    repo_for_subscribing_to_trial_json_file_changes.subscribe(observer)

    # /trials/trial1.json
    json_file = "cache_folder/trail.json"
    f = open(json_file)
    data = json.load(f)

    # every 2 seconds, notify the observers with the latest updated trials (the ones that have been updated recently)
    # notify a tuple of repo, trial: Tuple[HyperparamsRepository, Trial]

    hp_repo_for_automl = HyperparamsJSONRepository(
        cache_folder=os.path.join(tmpdir, 'cache'),
        best_retrained_model_folder=os.path.join(tmpdir, 'best')
    )
    auto_ml = _create_automl_loop(hp_repo_for_automl)

    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 4

    # When
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # TODO: assert tensorboard has written some files in the right directories
    validation_values = data['validation_splits'][0]["metric_results"]["main"]['validation_values']
    train_values = data['validation_splits'][0]["metric_results"]["main"]['train_values']

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train_values'
    validation_log_dir = 'logs/gradient_tape/' + current_time + '/validation_values'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
    EPOCH = len(validation_values)
    for epoch in range(EPOCH):
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_values[epoch], step=epoch)

        with validation_summary_writer.as_default():
            tf.summary.scalar('loss', validation_values[epoch], step=epoch)
    assert True
