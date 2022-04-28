import pandas as pd
import numpy as np
import tensorflow as tf

def format_input(dataset, num_inputs):
    data = []
    cols = []
    for peak in dataset:
        peak = ([(x+2)/4 for x in peak[0]], peak[1])
        if peak[1] == 'N':
            inputs = peak[0] + [0]
        else:
            inputs = peak[0] + [1]

        data.append(inputs)

    for i in range(num_inputs):
        cols.append(str(i))

    cols.append('labels')
    df = pd.DataFrame(data, columns = cols)
    return df

def get_totals(labels, pred):
    tot = [0,0,0,0]
    for i in range(len(pred)):
        a = pred[i]
        b = list(labels)[i]
        if a==1 and b==1:
            tot[0]+=1
        elif a==0 and b==0:
            tot[1]+=1
        elif a==1 and b==0:
            tot[2]+=1
        elif a==0 and b==1:
            tot[3]+=1

    return tot

def tensorflow_ml(training_data, test_data, num_inputs):
    # takes training dataset of a dataframe of peaks data
    dataset = format_input(training_data, num_inputs)
    data = dataset.drop(labels='labels', axis=1)
    labels = dataset['labels']
    print(data.shape)

    test_dataset = format_input(test_data, num_inputs)
    test_data = test_dataset.drop(labels='labels', axis=1)
    test_labels = test_dataset['labels']

    data = np.array(data)
    test_data = np.array(test_data)

    model = tf.keras.Sequential([
      #normalise,
      tf.keras.layers.Dense(135, activation='relu'),
      tf.keras.layers.Dense(2)
    ])

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer =tf.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])

    model.fit(data, labels, epochs=10)

    print(model.summary())

    _, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_data)

    pred = []
    for prediction in predictions:
        pred.append(np.argmax(prediction))

    totals = get_totals(test_labels, pred)
    print(f'True classifications: {list(test_labels)}')
    print(f'Predicted classifications: {pred}')
    print(f'True positives: {totals[0]}, True negatives: {totals[1]}, False positives: {totals[2]}, ' +
        f'False negatives: {totals[3]}')
