"""Implements a Deep Belief Network."""
from dbm import *
from tarfile import TUREAD

class DBN(DBM):

  def __init__(self, net, t_op=None, e_op=None):
    rbm, upward_net, downward_net, junction_layers = DBN.SplitDBN(net)
    #this help to separate top rbm with whole model 
    r_t_op = CopyOperation(t_op)
    self.rbm = DBM(rbm, r_t_op, e_op)
    self.upward_net = NeuralNet(upward_net, t_op, e_op)
    self.downward_net = NeuralNet(downward_net, t_op, e_op)
    self.junction_layers = junction_layers
    # add all_layers for future use
    self.all_layers = net.layer
    self.net = net
    self.t_op = t_op
    self.e_op = e_op
    self.verbose = self.rbm.verbose
    self.batchsize = self.t_op.batchsize
    
    
  def CopyModelToCPU(self):
    self.rbm.CopyModelToCPU()
    self.upward_net.CopyModelToCPU()
    self.downward_net.CopyModelToCPU()

  def DeepCopy(self):
    return CopyModel(self.rbm.net)

  def Show(self):
    """Visualize the state of the layers and edges in the network."""
    self.rbm.Show()
    self.upward_net.Show()
    self.downward_net.Show()

  def PrintNetwork(self):
    print 'RBM:'
    self.rbm.PrintNetwork()
    print 'Up:'
    self.upward_net.PrintNetwork()
    print 'Down:'
    self.downward_net.PrintNetwork()

  def ExchangeGlobalInfo(self):
    for layer in self.rbm.layer:
      layer.GetGlobalInfo(self)
    for edge in self.rbm.edge:
      edge.GetGlobalInfo(self)

  @staticmethod
  def SplitDBN(net):
    # net = ReadModel(dbn_file)
    rbm = deepnet_pb2.Model()
    rbm.CopyFrom(net)
    rbm.name = '%s_rbm' % net.name
    rbm.model_type = deepnet_pb2.Model.DBM
    
    directed_edges = []
    undirected_edges = []
    layer1 = set()  # Layers that are touched by directed edges.
    layer2 = set()  # Layers that are touched by undirected edges.
    for e in net.edge:
      if e.directed:
        directed_edges.append(e)
        layer1.add(e.node1)
        layer1.add(e.node2)
      else:
        undirected_edges.append(e)
        layer2.add(e.node1)
        layer2.add(e.node2)

    junction_layers = list(layer1.intersection(layer2))
    
    # CONTRUCT RBM.
    del rbm.edge[:]
    for e in undirected_edges:
      rbm.edge.extend([e])

    del rbm.layer[:]
    for node in list(layer2):
      l = next(l for l in net.layer if l.name == node)
      layer = rbm.layer.add()
      layer.CopyFrom(l)
      if node in junction_layers:
        layer.is_input = True
        del layer.param[:]
        for p in l.param:
          if p.name == 'bias':
            continue
          elif p.name == 'bias_generative':
            p_copy = layer.param.add()
            p_copy.CopyFrom(p)
            p_copy.name = 'bias'
          else:
            layer.param.extend([p])

    # CONSTRUCT DOWNWARD NET.
    down_net = deepnet_pb2.Model()
    down_net.CopyFrom(net)
    down_net.name = '%s_downward_net' % net.name
    down_net.model_type = deepnet_pb2.Model.FEED_FORWARD_NET

    del down_net.edge[:]
    for e in directed_edges:
      down_net.edge.extend([e])

    del down_net.layer[:]
    for node in list(layer1):
      l = next(l for l in net.layer if l.name == node)
      layer_down = down_net.layer.add()
      layer_down.CopyFrom(l)
      if l.is_input:
        layer_down.is_input = False
      if node in junction_layers:
        layer_down.is_input = True
      del layer_down.param[:]
      for p in l.param:
        if p.name == 'bias':
          continue
        elif p.name == 'bias_generative':
          p_copy = layer_down.param.add()
          p_copy.CopyFrom(p)
          p_copy.name = 'bias'
        else:
          layer_down.param.extend([p])

    # CONSTRUCT UPWARD NET.
    up_net = deepnet_pb2.Model()
    up_net.CopyFrom(net)
    up_net.name = '%s_upward_net' % net.name
    up_net.model_type = deepnet_pb2.Model.FEED_FORWARD_NET
    del up_net.edge[:]
    for e in directed_edges:
      e_up = DBN.ReverseEdge(e)
      up_net.edge.extend([e_up])
    del up_net.layer[:]
    for node in list(layer1):
      l = next(l for l in net.layer if l.name == node)
      layer_up = up_net.layer.add()
      layer_up.CopyFrom(l)
      del layer_up.param[:]
      for p in l.param:
        if p.name == 'bias_generative':
          continue
        else:
          layer_up.param.extend([p])

    return rbm, up_net, down_net, junction_layers

  @staticmethod
  def ReverseEdge(e):
    rev_e = deepnet_pb2.Edge()
    rev_e.CopyFrom(e)
    rev_e.node1 = e.node2
    rev_e.node2 = e.node1
    rev_e.up_factor = e.down_factor
    rev_e.down_factor = e.up_factor
    for p in rev_e.param:
      if p.name == 'weight':
        if p.initialization == deepnet_pb2.Parameter.PRETRAINED:
          p.transpose_pretrained = not p.transpose_pretrained
        elif p.mat:
          mat = ParameterAsNumpy(p).T
          p.mat = NumpyAsParameter(mat)
          del p.dimensions
          for dim in mat.shape:
            p.dimensions.add(dim)
    return rev_e

  def LoadModelOnGPU(self, *args, **kwargs):
    self.rbm.LoadModelOnGPU(*args, **kwargs)
    self.upward_net.LoadModelOnGPU(*args, **kwargs)
    self.downward_net.LoadModelOnGPU(*args, **kwargs)
    self.TieUpNets()

  def TieUpNets(self):
    # Tie up nets.
    for layer_name in self.junction_layers:
      rbm_layer = next(l for l in self.rbm.layer if l.name == layer_name)
      up_layer = next(l for l in self.upward_net.layer if l.name == layer_name)
      down_layer = next(l for l in self.downward_net.layer if l.name == layer_name)
      rbm_layer.data = up_layer.state
      down_layer.data = rbm_layer.state

  def ResetBatchsize(self, batchsize):
    self.batchsize = batchsize
    self.rbm.ResetBatchsize(batchsize)
    self.upward_net.ResetBatchsize(batchsize)
    self.downward_net.ResetBatchsize(batchsize)
    self.TieUpNets()

  def SetUpData(self, *args, **kwargs):
    self.upward_net.SetUpData(*args, **kwargs)
    self.train_data_handler = self.upward_net.train_data_handler
    self.validation_data_handler = self.upward_net.validation_data_handler
    self.test_data_handler = self.upward_net.test_data_handler

  def GetBatch(self, handler=None):
    if handler:
      data_list = handler.Get()
      if data_list[0].shape[1] != self.batchsize:
        self.ResetBatchsize(data_list[0].shape[1])
      for i, layer in enumerate(self.upward_net.datalayer):
        layer.SetData(data_list[i])
    for layer in self.upward_net.tied_datalayer:
      layer.SetData(layer.tied_to.data)

  def TrainOneBatch(self, step):
    if self.t_op.optimizer == deepnet_pb2.Operation.UPDOWN:
        return self.UpDownFineTuning(step=step, gibbs=False)
    elif self.t_op.optimizer == deepnet_pb2.Operation.TP:
        return self.TargetPropFineTuning(step=step)
    elif self.t_op.optimizer == deepnet_pb2.Operation.VI:
        #return self.MinVIFineTuning(step=step)
        return self.MinVIFineTuning_v2(step=step, forceWeightsTied=False)
    elif self.t_op.optimizer == deepnet_pb2.Operation.VI_ACCUM:
        return self.MinVIFineTuning_v2(step=step, forceWeightsTied=False)
    else:
        self.upward_net.ForwardPropagate(train=True, step=step)
        return self.rbm.TrainOneBatch(step)
 
  def PositivePhase(self, train=False, evaluate=False, step=0):
    self.upward_net.ForwardPropagate(train=train, step=step)
    return self.rbm.PositivePhase(train=train, evaluate=evaluate, step=step)
    # self.downward_net.ForwardPropagate(train=train, step=step)

  def NegativePhase(self, *args, **kwargs):
    return self.rbm.NegativePhase(*args, **kwargs)

  def Inference(self, steps, layernames, unclamped_layers, output_dir, memory='1G', dataset='test', method='gibbs'):
    layers_to_infer = [self.GetLayerByName(l, down=True) for l in layernames]
    layers_to_unclamp = [self.GetLayerByName(l) for l in unclamped_layers]

    numdim_list = [layer.state.shape[0] for layer in layers_to_infer]
    upward_net_unclamped_inputs = []
    for l in layers_to_unclamp:
      l.is_input = False
      l.is_initialized = True
      if l in self.rbm.layer:
        self.rbm.pos_phase_order.append(l)
      else:
        upward_net_unclamped_inputs.append(l)

    if dataset == 'train':
      datagetter = self.GetTrainBatch
      if self.train_data_handler is None:
        return
      numbatches = self.train_data_handler.num_batches
      size = numbatches * self.train_data_handler.batchsize
    elif dataset == 'validation':
      datagetter = self.GetValidationBatch
      if self.validation_data_handler is None:
        return
      numbatches = self.validation_data_handler.num_batches
      size = numbatches * self.validation_data_handler.batchsize
    elif dataset == 'test':
      datagetter = self.GetTestBatch
      if self.test_data_handler is None:
        return
      numbatches = self.test_data_handler.num_batches
      size = numbatches * self.test_data_handler.batchsize
    dw = DataWriter(layernames, output_dir, memory, numdim_list, size)

    gibbs = method == 'gibbs'
    mf = method == 'mf'

    for batch in range(numbatches):
      sys.stdout.write('\r%d' % (batch + 1))
      sys.stdout.flush()
      datagetter()
      for l in upward_net_unclamped_inputs:
        l.data.assign(0)
      self.upward_net.ForwardPropagate()
      for node in self.rbm.node_list:
        if node.is_input or node.is_initialized:
        # Here, it has been announced during the setupData phrase that later.data = upward_net.state
          node.GetData()
          if gibbs:
            node.sample.assign(node.state)
        else:
          node.ResetState(rand=False)
      for i in range(steps):
        for node in self.rbm.pos_phase_order:
          self.ComputeUp(node, use_samples=gibbs)
          if gibbs:
            node.Sample()
      self.downward_net.ForwardPropagate()
      output = [l.state.asarray().T for l in layers_to_infer]
      dw.Submit(output)
    sys.stdout.write('\n')
    size = dw.Commit()
    return size[0]

  def GetLayerByName(self, layername, down=False):
    layer = self.rbm.GetLayerByName(layername)
    if layer is None:
      if down:
        layer = self.downward_net.GetLayerByName(layername)
      else:
        layer = self.upward_net.GetLayerByName(layername)
    return layer
   
  def GetSampleFromLayers(self, layernames, mc_steps, sampleNum, output_dir, all=False, memory='1G', dataset='validation', method='gibbs'):
    layers_to_sample = []
    back_prop_layers = []
    
    temp_pos_phase_order = []
    temp_pos_phase_order.append(self.rbm.pos_phase_order[0])     
    # force add the input layers into the phase_order for sampling                             
    for node in self.rbm.input_datalayer:
        temp_pos_phase_order.append(node) 
    for l in self.upward_net.layer:
        if l.is_input: 
          back_prop_layers.append(l.name) 
    
    input_layers = [self.GetLayerByName(l, down=True) for l in back_prop_layers]               
                
    if all:
      layers_to_sample = [self.GetLayerByName(l, down=True) for l in self.net.layer]
    else:
      layers_to_sample = [self.GetLayerByName(l, down=True) for l in layernames]               
      
      
    numdim_list = [layer.state.shape[0] for layer in layers_to_sample]
    numdim_list_projection = [layer.state.shape[0] for layer in input_layers]
    
    if dataset == 'train':
      datagetter = self.GetTrainBatch
      if self.train_data_handler is None:
        return
      batchsize = self.train_data_handler.batchsize
      size = sampleNum * self.train_data_handler.batchsize
    elif dataset == 'validation':
      datagetter = self.GetValidationBatch
      if self.validation_data_handler is None:
        return
      batchsize = self.validation_data_handler.batchsize
      size = sampleNum * self.validation_data_handler.batchsize
    elif dataset == 'test':
      datagetter = self.GetTestBatch
      if self.test_data_handler is None:
        return
      batchsize = self.test_data_handler.batchsize
      size = sampleNum * self.test_data_handler.batchsize
    dw = DataWriter(layernames, output_dir, memory, numdim_list, size)
    
    joint_project = []
    for ln in back_prop_layers:
      joint_project.append(ln + '_joint')
      
    dw_back_propped_joint = DataWriter(joint_project, output_dir, memory, numdim_list_projection, size)  
    dw_back_propped = DataWriter(back_prop_layers, output_dir, memory, numdim_list_projection, size)
    
    gibbs = method == 'gibbs'
    datagetter()
      # forwardPropagate assign a good initial states for layers 
    self.upward_net.ForwardPropagate()
    for node in self.rbm.node_list:
        if node.is_input or node.is_initialized:
        # Here, it has been announced during the setupData phrase that later.data = upward_net.state
          node.GetData()
          if gibbs:
            node.sample.assign(node.state)
        else:
          node.ResetState(rand=False)
    # here, we are supposed to use samples not the probability when we do the sampling 
        
    for i in range(mc_steps):
        for node in temp_pos_phase_order:
          self.ComputeUp(node, use_samples=gibbs, compute_input=True)
          if gibbs:
            node.Sample()
            
    # suppose after mc_steps the Markov Chain has reach the equilibrium state  
    inter_space = [0, 1, 5, 10, 20, 50, 100, 200, 500]; 
    interpolation_neigbors = [];
    interpolation_neigbors_joint = [];     
    for i in range(sampleNum):
        for node in temp_pos_phase_order:
          self.ComputeUp(node, use_samples=gibbs, compute_input=True)
          if gibbs:
            node.Sample()
        # samples generated from wanted layers    
        output = [l.sample.asarray().T for l in layers_to_sample] 
        dw.Submit(output)    
        if i in inter_space:
            interpolation_neigbors.append(np.copy(output[0].T))
            interpolation_neigbors.append(np.copy(output[1].T))
            interpolation_neigbors_joint.append(np.copy(output[2].T)) 
            
        # these are the samples back-propped from top joint layer
        for node in temp_pos_phase_order:
          if node in self.rbm.pos_phase_order:
              continue
          # one step more for projection because downward_net do not contains the top layer of rbm, here I choose to use state rather than sample
          self.ComputeUp(node, use_samples=True, compute_input=True)
        
        self.downward_net.ForwardPropagate()
        projected_samples_joint = [l.state.asarray().T for l in input_layers]
        dw_back_propped_joint.Submit(projected_samples_joint)
        
        # these are the samples back-propped from the junction layers  
        for node in temp_pos_phase_order:
          if node in self.rbm.pos_phase_order:
              continue  
          node.state.assign(node.sample)
           
        self.downward_net.ForwardPropagate()
        projected_samples = [l.state.asarray().T for l in input_layers]     
              
        dw_back_propped.Submit(projected_samples)
       
    # project the interpolation to 
    self.Interpolation(interpolation_neigbors_joint, interpolation_neigbors, temp_pos_phase_order, output_dir, batchsize, numdim_list_projection, input_layers)
    
    
     
    sys.stdout.write('\n')
    size = dw.Commit()
    dw_back_propped_joint.Commit()
    dw_back_propped.Commit()
    
    return size[0]
   
  def Interpolation(self, interpolation_neigbors_joint, interpolation_neigbors, temp_pos_phase_order, output_dir, batchsize, numdim_list_projection, input_layers):
        num = len(interpolation_neigbors_joint);
        name = ['interpolation_text', 'interpolation_image']
        name_joint = ['interpolation_text_joint', 'interpolation_image_joint']
        dw = DataWriter(name, output_dir, '20G', numdim_list_projection, (num - 1) * batchsize) 
        dw_joint = DataWriter(name_joint, output_dir, '20G', numdim_list_projection, (num - 1) * batchsize) 
        for i in range(num - 1):
            sample_joint = (interpolation_neigbors_joint[0] + interpolation_neigbors_joint[i + 1]) / 2
            sample_image = (interpolation_neigbors[0] + interpolation_neigbors[2 * i + 2]) / 2
            sample_text = (interpolation_neigbors[1] + interpolation_neigbors[2 * i + 3]) / 2
            sample_mat_joint = cm.CUDAMatrix(sample_joint)
            sample_mat_image = cm.CUDAMatrix(sample_image)
            sample_mat_text = cm.CUDAMatrix(sample_text)
            self.rbm.pos_phase_order[0].sample.assign(sample_mat_joint)
            for node in temp_pos_phase_order:
              if node in self.rbm.pos_phase_order:
                  continue
              # one step more for projection because downward_net do not contains the top layer of rbm, here I choose to use state rather than sample
              self.ComputeUp(node, use_samples=True, compute_input=True)
            
            self.downward_net.ForwardPropagate()
            projected_samples_joint = [l.state.asarray().T for l in input_layers]
            dw_joint.Submit(projected_samples_joint)
            
            for node in temp_pos_phase_order:
                if node.name == 'image_hidden2' :
                    node.state.assign(sample_mat_image)
                if node.name == 'text_hidden2' :
                    node.state.assign(sample_mat_text)    
                    
            
            self.downward_net.ForwardPropagate()
            projected_samples = [l.state.asarray().T for l in input_layers]
            dw.Submit(projected_samples)
            
            sample_mat_joint.free_device_memory()
            sample_mat_image.free_device_memory()
            sample_mat_text.free_device_memory()
  
        
           
      
      
      
      
      
      
        
      
            
      

