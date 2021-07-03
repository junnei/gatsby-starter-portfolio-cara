function modelCode(props){
    return `class Model(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    `+(modelLayer(props.model))+`

  def forward(self, x):
    `+(modelForward(props.model))+`
    return x
`
};
/*
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)

    output = F.log_softmax(x, dim=1)
*/
function modelForward(layers){

  var modelNum={
    linear:0,
    conv2d:0,
    dropout2d:0
  };
  
  function layerType(type, props){
    var tempNum = 0;
    if(type === "linear"){
      modelNum.linear += 1;
      tempNum = modelNum.linear;
    }
    else if(type === "conv2d"){
      modelNum.conv2d += 1;
      tempNum = modelNum.conv2d;
    }
    else if(type === "dropout2d"){
      modelNum.dropout2d += 1;
      tempNum = modelNum.dropout2d;
    }

    if(type === "linear"
    || type === "conv2d"
    || type === "dropout2d"
    ){
      return 'x = self.'+type+tempNum+'(x)';
    }
    else if(type === "relu"){
      return 'x = F.'+type+'(x)';
    }
    else if(type === "max_pool2d"){
      return 'x = F.'+type+'(x, '+props.x+')';
    }
    else if(type === "flatten"){
      return 'x = torch.'+type+'(x, '+props.dim+')';
    }
    else if(type === "log_softmax"){
      return 'x = F.'+type+'(x, dim='+props.dim+')';
    }
  }

  return layers.map(x => layerType(x.type, x.props)).join(`
    `)+`
    `;
};

function modelLayer(layers){

  var modelNum={
    linear:0,
    conv2d:0,
    dropout2d:0
  };

  function layerType(type){
    var tempNum = 0;
    if(type === "linear"){
      modelNum.linear += 1;
      tempNum = modelNum.linear;
    }
    else if(type === "conv2d"){
      modelNum.conv2d += 1;
      tempNum = modelNum.conv2d;
    }
    else if(type === "dropout2d"){
      modelNum.dropout2d += 1;
      tempNum = modelNum.dropout2d;
    }

    function capitalize(str){
      return str.charAt(0).toUpperCase() + str.slice(1);
    }

    return 'self.'+type+tempNum+' = nn.'+capitalize(type);
  }

  return layers.map(x => 
    (x.type === "linear"
    || x.type === "conv2d"
    || x.type === "dropout2d") ?
      layerType(x.type)+layerProps(x.type, x.props)+`
    `
      :
      null
      ).join('');
};

function layerProps(type, props){
  if(type === "linear"){
    return linearProps(props);
  }
  if(type === "dropout2d"){
    return dropout2dProps(props);
  }
  else if(type === "conv2d"){
    return conv2dProps(props);
  }

  function linearProps(props){
    return '('+props.in+', '+props.out+')';
  }
  function conv2dProps(props){
    return '('+props.in+', '+props.out+', '+props.kernel+(props.stride?(', '+props.stride):(''))+')';
  }
  function dropout2dProps(props){
    return '('+props.p+')';
  }
};


function train_lib(libs){
  return `import torch
import torch.nn as nn
import torch.nn.functional as F
`
};

function trainCode(props){
    return train_lib(props)+
props.data.imageSizeX+`,`+props.data.imageSizeY+
((props.data.imageSizeX==="244") ?
`from torch.utils.tensorboard import SummaryWriter`
:
``)
+
`
import matplotlib.pyplot as plt
`};


function files(props) {

    const file = {
    "model.py": {
        name: "model.py",
        language: "python",
        value: modelCode(props)
    },
    "train.py": {
        name: "train.py",
        language: "python",
        value: trainCode(props)
    }
    };

    return file[props.fileName];
}

export default files;
