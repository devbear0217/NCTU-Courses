import os
import cv2
import timeit
import numpy as np
import tensorflow as tf

camera = cv2.VideoCapture(0)

label_lines = [line.rstrip() for line
               in tf.gfile.GFile('retrained_labels.txt')]


def grab_video_feed():
    grabbed, frame = camera.read()
    return frame if grabbed else None


def initial_setup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = timeit.default_timer()

    with tf.gfile.FastGFile('retrained_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    print('Took {} seconds to unpersist the graph'.format(timeit.default_timer() - start_time))


initial_setup()

with tf.Session() as sess:
    start_time = timeit.default_timer()
    softmax_tensor = sess.graph.get_tensor_by_name('input:0')

    print('Took {} seconds to feed data to graph'.format(timeit.default_timer() - start_time))

    while True:
        frame = grab_video_feed()

        if frame is None:
            raise SystemError('Issue grabbing the frame')

        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)

        cv2.imshow('Main', frame)

        numpy_frame = np.asarray(frame)
        numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        numpy_final = np.expand_dims(numpy_frame, axis=0)

        start_time = timeit.default_timer()
        predictions = sess.run(softmax_tensor, {'input:0': numpy_final})

        print('Took {} seconds to perform prediction'.format(timeit.default_timer() - start_time))

        start_time = timeit.default_timer()

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        print('Took {} seconds to sort the predictions'.format(timeit.default_timer() - start_time))
        print('********* Session Ended *********')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sess.close()
            break

camera.release()
cv2.destroyAllWindows()
