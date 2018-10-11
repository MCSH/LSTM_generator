Config = {
    'data_set': 'input.txt',
    'model_name': 'saved_model.h5',
    'LSTM_size': 64,
    'LSTM_count': 2,
    'sample_size': 400,
    'epoch_count': 60,
    'batch_size': 128,
    'save_iter': 1, # How many iterations should it pass before saving the model?
    'print_iter': 3, # How many interation should it pass before printing sample?
    'save_on_finish': True,
    'diversity': [0.2, 0.5, 1.0, 1.2]
}
