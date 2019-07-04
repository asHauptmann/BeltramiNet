''' Class file for Unet for DeepDbar
# Written 04. October 2018 by Andreas Hauptmann, UCL'''

import tensorflow as tf
import numpy
import h5py
import os 
from os.path import exists
FLAGS = None

''' !!!SET Directory for Tensorboard!!! '''
tensorboardDefaultDir =  '/scratch1/NOT_BACKED_UP/ahauptma/Dropbox/tensorboard/CMR_test/'


#Loading part

def extract_images(filename,imageName):
  """Extract the images into a 4D numpy array [index, y, x, depth]."""
  fData = h5py.File(filename,'r')
  inData = fData.get(imageName)  
      
  
  num_images = inData.shape[0]
  rows = inData.shape[1]
  cols = inData.shape[2]
  print(num_images, rows, cols)
  data = numpy.array(inData)
    
  data = data.reshape(num_images, rows, cols, 1)
  return data

class DataSet(object):

  def __init__(self, images, true):
    """Construct a DataSet"""

    self._num_examples = images.shape[0]

    #This is somewhat redundant
    images = images.reshape(images.shape[0],
                            images.shape[1],images.shape[2])
    true = true.reshape(true.shape[0],
                            true.shape[1],true.shape[2])

    self._images = images
    self._true = true
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def true(self):
    return self._true

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._true = self._true[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._true[start:end]


def read_data_sets(FileNameTrain,FileNameTest):
  class DataSets(object):
    pass
  data_sets = DataSets()

  TRAIN_SET = FileNameTrain
  TEST_SET  = FileNameTest
  IMAGE_NAME = 'imagesRecon'
  TRUE_NAME  = 'imagesTrue'
  
  print('Start loading data')  
  train_images = extract_images(TRAIN_SET,IMAGE_NAME)
  train_true   = extract_images(TRAIN_SET,TRUE_NAME)

  
  test_images = extract_images(TEST_SET,IMAGE_NAME)
  test_true   = extract_images(TEST_SET,TRUE_NAME)


  data_sets.train = DataSet(train_images, train_true)
  data_sets.test = DataSet(test_images, test_true)

  return data_sets

def create_data_sets(test_images):
  class DataSets(object):
    pass
  data_sets = DataSets()

  data_sets.test = DataSet(test_images, test_images)

  return data_sets


def resUnet(x,imSize,bSize):

   

  x_image = tf.reshape(x, [-1, imSize[1],imSize[2], 1])
  
  
  x_image1 = tf.contrib.layers.conv2d(x_image, 32,5)
  x_image1 = tf.contrib.layers.conv2d(x_image1, 32,5)
  
  
  
  # Maxpool layer 1
  x_image2 = max_pool_2x2(x_image1)   
  x_image2 = tf.contrib.layers.conv2d(x_image2, 64,5)
  x_image2 = tf.contrib.layers.conv2d(x_image2, 64,5)
  
  # Maxpool layer 2
  x_image3 = max_pool_2x2(x_image2)   
  x_image3 = tf.contrib.layers.conv2d(x_image3, 128,5)
  x_image3 = tf.contrib.layers.conv2d(x_image3, 128,5)
  
  
  # Maxpool layer 3
  x_image4 = max_pool_2x2(x_image3)   
  x_image4 = tf.contrib.layers.conv2d(x_image4, 256,5)
  x_image4 = tf.contrib.layers.conv2d(x_image4, 256,5)
    

  
  #Maxpool layer 4
  x_image5 = max_pool_2x2(x_image4)   
  x_image5 = tf.contrib.layers.conv2d(x_image5, 512,5)
  x_image5 = tf.contrib.layers.conv2d(x_image5, 512,5)  

  
  sizeX=int(imSize[1]/8)
  sizeY=int(imSize[2]/8)
  #Upsample and concat ------ B2
  W_TconvPrioD3 = weight_variable([5, 5, 256, 512]) #Ouput, Input channels
  b_TconvPrioD3 = bias_variable([256])
  x_imageUp4 = tf.nn.relu(conv2d_trans(x_image5, W_TconvPrioD3,[bSize,sizeX,sizeY,256]) + b_TconvPrioD3)
  
  x_imageUp4 = tf.concat([x_image4,x_imageUp4],3)
  x_imageUp4 = tf.contrib.layers.conv2d(x_imageUp4, 256,5)
  x_imageUp4 = tf.contrib.layers.conv2d(x_imageUp4, 256,5)  
  
  
  sizeX=int(imSize[1]/4)
  sizeY=int(imSize[2]/4)
  #Upsample and concat ------ B2
  W_TconvPrioC3 = weight_variable([5, 5, 128, 256]) #Ouput, Input channels
  b_TconvPrioC3 = bias_variable([128])
  x_imageUp3 = tf.nn.relu(conv2d_trans(x_imageUp4, W_TconvPrioC3,[bSize,sizeX,sizeY,128]) + b_TconvPrioC3)
    

  x_imageUp3 = tf.concat([x_image3,x_imageUp3],3)
  x_imageUp3 = tf.contrib.layers.conv2d(x_imageUp3, 128,5)
  x_imageUp3 = tf.contrib.layers.conv2d(x_imageUp3, 128,5)  



  sizeX=int(imSize[1]/2)
  sizeY=int(imSize[2]/2)
  #Upsample and concat ------ B2
  W_TconvPrioB3 = weight_variable([5, 5, 64, 128]) #Ouput, Input channels
  b_TconvPrioB3 = bias_variable([64])
  x_imageUp2 = tf.nn.relu(conv2d_trans(x_imageUp3, W_TconvPrioB3,[bSize,sizeX,sizeY,64]) + b_TconvPrioB3)
  
  
  x_imageUp2 = tf.concat([x_image2,x_imageUp2],3)
  x_imageUp2 = tf.contrib.layers.conv2d(x_imageUp2, 64,5)
  x_imageUp2 = tf.contrib.layers.conv2d(x_imageUp2, 64,5)
  
  #Upsample and concat ------ B2
  W_TconvPrioA3 = weight_variable([5, 5, 32, 64]) #Ouput, Input channels
  b_TconvPrioA3 = bias_variable([32])
  x_imageUp1 = tf.nn.relu(conv2d_trans(x_imageUp2, W_TconvPrioA3,[bSize,imSize[1],imSize[2],32]) + b_TconvPrioA3)
  
  x_imageUp1 = tf.concat([x_image1,x_imageUp1],3)
  
  x_imageUp1 = tf.contrib.layers.conv2d(x_imageUp1, 32,5)
  x_imageUp1 = tf.contrib.layers.conv2d(x_imageUp1, 32,5)
  
  x_imageUp1 = tf.contrib.layers.conv2d(x_imageUp1, 1,5,activation_fn=None)
    
  x_sum       = tf.add(x_imageUp1,x_image)
  
  x_update = tf.nn.relu(x_sum)
  x_update = tf.reshape(x_sum, [-1, imSize[1],imSize[2]])
  
  return x_update



def Unet(x,imSize,bSize):

  x_image = tf.reshape(x, [-1, imSize[1],imSize[2], 1])
  
  
  x_image1 = tf.contrib.layers.conv2d(x_image, 32,5)
  x_image1 = tf.contrib.layers.conv2d(x_image1, 32,5)
  
  
  
  # Maxpool layer 1
  x_image2 = max_pool_2x2(x_image1)   
  x_image2 = tf.contrib.layers.conv2d(x_image2, 64,5)
  x_image2 = tf.contrib.layers.conv2d(x_image2, 64,5)
  
  # Maxpool layer 2
  x_image3 = max_pool_2x2(x_image2)   
  x_image3 = tf.contrib.layers.conv2d(x_image3, 128,5)
  x_image3 = tf.contrib.layers.conv2d(x_image3, 128,5)
  
  
  # Maxpool layer 3
  x_image4 = max_pool_2x2(x_image3)   
  x_image4 = tf.contrib.layers.conv2d(x_image4, 256,5)
  x_image4 = tf.contrib.layers.conv2d(x_image4, 256,5)
  
  #Maxpool layer 4
  x_image5 = max_pool_2x2(x_image4)   
  x_image5 = tf.contrib.layers.conv2d(x_image5, 512,5)
  x_image5 = tf.contrib.layers.conv2d(x_image5, 512,5)  

  
  sizeX=int(imSize[1]/8)
  sizeY=int(imSize[2]/8)
  #Upsample and concat ------ B2
  W_TconvPrioD3 = weight_variable([5, 5, 256, 512]) #Ouput, Input channels
  b_TconvPrioD3 = bias_variable([256])
  x_imageUp4 = tf.nn.relu(conv2d_trans(x_image5, W_TconvPrioD3,[bSize,sizeX,sizeY,256]) + b_TconvPrioD3)
  
  x_imageUp4 = tf.concat([x_image4,x_imageUp4],3)
  x_imageUp4 = tf.contrib.layers.conv2d(x_imageUp4, 256,5)
  x_imageUp4 = tf.contrib.layers.conv2d(x_imageUp4, 256,5)  
  
  
  sizeX=int(imSize[1]/4)
  sizeY=int(imSize[2]/4)
  #Upsample and concat ------ B2
  W_TconvPrioC3 = weight_variable([5, 5, 128, 256]) #Ouput, Input channels
  b_TconvPrioC3 = bias_variable([128])
  x_imageUp3 = tf.nn.relu(conv2d_trans(x_imageUp4, W_TconvPrioC3,[bSize,sizeX,sizeY,128]) + b_TconvPrioC3)
    

  x_imageUp3 = tf.concat([x_image3,x_imageUp3],3)
  x_imageUp3 = tf.contrib.layers.conv2d(x_imageUp3, 128,5)
  x_imageUp3 = tf.contrib.layers.conv2d(x_imageUp3, 128,5)  


  sizeX=int(imSize[1]/2)
  sizeY=int(imSize[2]/2)
  #Upsample and concat ------ B2
  W_TconvPrioB3 = weight_variable([5, 5, 64, 128]) #Ouput, Input channels
  b_TconvPrioB3 = bias_variable([64])
  x_imageUp2 = tf.nn.relu(conv2d_trans(x_imageUp3, W_TconvPrioB3,[bSize,sizeX,sizeY,64]) + b_TconvPrioB3)
  
  
  x_imageUp2 = tf.concat([x_image2,x_imageUp2],3)
  x_imageUp2 = tf.contrib.layers.conv2d(x_imageUp2, 64,5)
  x_imageUp2 = tf.contrib.layers.conv2d(x_imageUp2, 64,5)
  
  #Upsample and concat ------ B2
  W_TconvPrioA3 = weight_variable([5, 5, 32, 64]) #Ouput, Input channels
  b_TconvPrioA3 = bias_variable([32])
  x_imageUp1 = tf.nn.relu(conv2d_trans(x_imageUp2, W_TconvPrioA3,[bSize,imSize[1],imSize[2],32]) + b_TconvPrioA3)
  
  x_imageUp1 = tf.concat([x_image1,x_imageUp1],3)
  
  x_imageUp1 = tf.contrib.layers.conv2d(x_imageUp1, 32,5)
  x_imageUp1 = tf.contrib.layers.conv2d(x_imageUp1, 32,5)
  
  x_imageUp1 = tf.contrib.layers.conv2d(x_imageUp1, 1,5)
    
#  x_sum       = tf.add(x_imageUp1,x_image)
  
#  x_update = tf.nn.relu(x_sum)
  x_update = tf.reshape(x_imageUp1, [-1, imSize[1],imSize[2]])
  
  return x_update

def conv2d(x, W):
  """conv3d returns a 3d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_trans(x, W, shape):
  """conv3d returns a 3d convolution layer with full stride."""
  return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.025)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.025, shape=shape)
  return tf.Variable(initial)


def psnr(x_result, x_true, name='psnr'):
    with tf.name_scope(name):
        maxval = tf.reduce_max(x_true) - tf.reduce_min(x_true)
        mse = tf.reduce_mean((x_result - x_true) ** 2)
        return 20 * log10(maxval) - 10 * log10(mse)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator        
 
    


def training(dataSet,bSize,trainIter,checkPointInterval,lossFunc,unetType,useTensorboard,experimentName,filePath):
# Import data
  
  lVal=1e-4 #Learning rate (could be changed)  
  
  sess = tf.InteractiveSession()    
  imSize=dataSet.train.true.shape
    
  # Create the model
  imag = tf.placeholder(tf.float32, [None, imSize[1],imSize[2]])
  true = tf.placeholder(tf.float32, [None, imSize[1],imSize[2]])
  
  
  # Build the graph for the deep net
  if unetType == 'Unet':
      display('Using Unet')
      x_out = Unet(imag,imSize,bSize)
  elif unetType == 'resUnet':
      display('Using Residual Unet')
      x_out = resUnet(imag,imSize,bSize)
  else:
      display('Not supported network')
      return
      

  with tf.name_scope('optimizer'):
         
         if lossFunc == 'l2_loss':
             loss = tf.nn.l2_loss(x_out-true)
             print('Using l2-loss')
         elif lossFunc == 'l1_loss':
             loss = tf.reduce_sum(tf.abs(x_out-true))
             print('Using l1-loss')
         else: 
             display('Not supported loss function')
             return
         #To rule out zero solution
         added_loss = -tf.scalar_mul(100.0,tf.minimum( tf.subtract(tf.norm(x_out),1.0),0.0))     
         learningRate=tf.constant(1e-4) # This is an init, can be changed
         train_step = tf.train.AdamOptimizer(learningRate).minimize(loss+added_loss)
    
    
  if(useTensorboard):
      with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('psnr', psnr(x_out, true))
    
    
        tf.summary.image('result', tf.reshape(x_out[1],[1, imSize[1], imSize[2], 1]) )
        tf.summary.image('true', tf.reshape(true[1],[1, imSize[1], imSize[2], 1]) )
        tf.summary.image('imag', tf.reshape(imag[1],[1, imSize[1], imSize[2], 1]) )
        
        
        merged_summary = tf.summary.merge_all()
        test_summary_writer, train_summary_writer = summary_writers('default',experimentName)
          
    
  
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  
  feed_test={imag: dataSet.test.images[0:bSize],
                 true: dataSet.test.true[0:bSize]}
                 

  for i in range(trainIter):
          
        batch = dataSet.train.next_batch(bSize)

        feed_train={imag: batch[0], true: batch[1],learningRate: lVal}
                 
        _, merged_summary_result_train = sess.run([train_step, merged_summary],
                                          feed_dict=feed_train)
        if i % 10 == 0:
            
            batchTest = dataSet.test.next_batch(bSize) 
                        
            feed_test={imag: batchTest[0], true: batchTest[1], learningRate: lVal}
            
            test_accuracy, test_result = sess.run([loss, x_out],
                                                          feed_dict=feed_test)
            
        
            loss_result, merged_summary_result = sess.run([loss, merged_summary],
                              feed_dict=feed_test)
        
            train_summary_writer.add_summary(merged_summary_result_train, i)
            test_summary_writer.add_summary(merged_summary_result, i)
        
            print('iter={}, loss={}'.format(i, loss_result))  
            
        
        if i % checkPointInterval == 0:
            checkPointName = filePath + experimentName + '_' + str(i)
            save_path = saver.save(sess, checkPointName)
            print("Model saved in file: %s" % save_path)
              #Save data
        
        
        
  checkPointName = filePath + experimentName + '_final'
  save_path = saver.save(sess, checkPointName)
  print("Model saved in file: %s" % save_path)
  return

def default_tensorboard_dir(name):
    tensorboard_dir = tensorboardDefaultDir 
    if not exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return tensorboard_dir


def summary_writers(name, expName , session=None):
    if session is None:
        session = tf.get_default_session()
    
    dname = default_tensorboard_dir(name)
    

    
    test_summary_writer = tf.summary.FileWriter(dname + '/test_' + expName, session.graph)
    train_summary_writer = tf.summary.FileWriter(dname + '/train_' + expName)
    
    return test_summary_writer, train_summary_writer


                        

def reconstruction(netPath,test_images,unetType):
  
    
  dataStruct = create_data_sets(test_images)  
  imSize=dataStruct.test.images.shape
  
  #Adjust, or get as input
  bSize=imSize[0]

  # Create the model
  imag = tf.placeholder(tf.float32, [None, imSize[1],imSize[2]])
  output = tf.placeholder(tf.float32, [None, imSize[1],imSize[2]])    
  

  # Build the graph for the deep net
  # Python does not have switch, so here we go with an elaborate elseif chain  
  # Build the graph for the deep net
  if unetType == 'Unet':
      display('Using Unet')
      x_out = Unet(imag,imSize,bSize)
  elif unetType == 'resUnet':
      display('Using Residual Unet')
      x_out = resUnet(imag,imSize,bSize)
  else:
      display('Not supported network')
      return test_images
      
          

  saver = tf.train.Saver()


  sess = tf.Session()
    
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, netPath)
  print("Model restored.")

  start=0
  end=bSize
      
  output = sess.run(x_out,feed_dict={imag: dataStruct.test.images[start:end]})

  print('Sample processed')
  
  tf.reset_default_graph()  
  sess.close()

  return output



