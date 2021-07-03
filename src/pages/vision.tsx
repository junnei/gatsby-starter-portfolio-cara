import React, {useRef, useState } from 'react'

import Editor from "@monaco-editor/react";
import files from "../components/files";

import JSZip from 'jszip';
import FileSaver from 'file-saver';

export default function Component () {

  const editorRef = useRef(null);

  function handleEditorDidMount(editor, monaco) {
    editorRef.current = editor; 
  }
  

  async function downloadZip() {
    const element = document.createElement("a");
    const initFile = inputs.fileName;

    async function setFile(file){
      setInputs({
        ...inputs,
        fileName : file
      });
      return new File([editorRef.current.getValue()], file, {type: 'text/plain'})
    };

    var zip = new JSZip();
    zip.file("model.py", await setFile("model.py"));
    zip.file("train.py", await setFile("train.py"));
    zip.generateAsync({ type: "blob" })
        .then(function callback(blob) {
            FileSaver.saveAs(blob, "MLCoke.zip");
        });

    setInputs({
      ...inputs,
      fileName : initFile
    });

  }

  function downloadFile() {
    const element = document.createElement("a");
    const file = new Blob([editorRef.current.getValue()], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = inputs.fileName;
    document.body.appendChild(element); // Required for this to work in FireFox
    element.click();
  }
  
  function showCode() {
    alert(editorRef.current.getValue());
  }

  function addLayer() {
    setInputs({
      ...inputs,
      layerNum: 0,
      model: [
        ...inputs.model,
        {
          layerNum: inputs.layerNum+1,
          type: inputs.currentLayer.type,
          props: inputs.currentLayer.props,
        }
      ],
    });
  }

  const [inputs, setInputs] = useState({
    layerNum: 0,
    fileName: "model.py",
    data: {
      imageSizeX: "244",
      imageSizeY: "244"
    },
    currentLayer: {
      type:"linear",
      props:{},
    },
    model: [
    ],

  })

  return (
   <>
     <form>
      <text>image size(x)</text>
      <input required
        placeholder="x"
        value={inputs.data.imageSizeX}
        onChange={(e) => setInputs({
          ...inputs,
          data : {
            imageSizeX: e.target.value,
            imageSizeY: inputs.data.imageSizeY,
          }
      })}
      />
      <text>image size(y)</text>
      <input required
        placeholder="y"
        value={inputs.data.imageSizeY}
        onChange={(e) => setInputs({
          ...inputs,
          data : {
            imageSizeX: inputs.data.imageSizeX,
            imageSizeY: e.target.value,
          }
      })}
      />
      <br></br>
      <select required
        name="layer_type"
        multiple size="6"
        onChange={(e) => setInputs({
          ...inputs,
          currentLayer : {
            type: e.target.options[e.target.selectedIndex].value,
            props:inputs.currentLayer.props,
          }
      })}
      >
        <optgroup label="nn.Module">
          <option value="conv2d">conv2d</option>
          <option value="dropout2d">dropout2d</option>
          <option value="linear">linear</option>
        </optgroup>

        <optgroup label="F.Module">
          <option value="relu">relu</option>
          <option value="max_pool2d">max_pool2d</option>
          <option value="log_softmax">log_softmax</option>
        </optgroup>

        <optgroup label="torch.Module">
          <option value="flatten">flatten</option>
        </optgroup>

      </select>
      
      {
      (inputs.currentLayer.type === 'max_pool2d')
      &&
      <div>
        <text>x</text>
        <input required
          placeholder="x"
          onChange={(e) => setInputs({
            ...inputs,
            currentLayer : {
              type: inputs.currentLayer.type,
              props: {
                ...inputs.currentLayer.props,
                x : e.target.value,
              }
            }
        })}
        />
      </div>
      }

      {
      (inputs.currentLayer.type === 'log_softmax'
      || inputs.currentLayer.type === 'flatten')
      &&
      <div>
        <text>dim</text>
        <input required
          placeholder="dim"
          onChange={(e) => setInputs({
            ...inputs,
            currentLayer : {
              type: inputs.currentLayer.type,
              props: {
                ...inputs.currentLayer.props,
                dim : e.target.value,
              }
            }
        })}
        />
      </div>
      }

      {
      (inputs.currentLayer.type === 'dropout2d')
      &&
      <div>
        <text>p</text>
        <input required
          placeholder="p"
          onChange={(e) => setInputs({
            ...inputs,
            currentLayer : {
              type: inputs.currentLayer.type,
              props: {
                ...inputs.currentLayer.props,
                p : e.target.value,
              }
            }
        })}
        />
      </div>
      }
      
      {
      (inputs.currentLayer.type === 'conv2d'
      ||
      inputs.currentLayer.type === 'linear'
      )
      &&
      <div>
        <text>in</text>
        <input required
          placeholder="in"
          onChange={(e) => setInputs({
            ...inputs,
            currentLayer : {
              type: inputs.currentLayer.type,
              props: {
                ...inputs.currentLayer.props,
                in : e.target.value,
              }
            }
        })}
        />
      </div>
      }

      {
      (inputs.currentLayer.type === 'conv2d'
      ||
      inputs.currentLayer.type === 'linear'
      )
      &&
      <div>
        <text>out</text>
        <input required
          placeholder="out"
          onChange={(e) => setInputs({
            ...inputs,
            currentLayer : {
              type: inputs.currentLayer.type,
              props: {
                ...inputs.currentLayer.props,
                out : e.target.value,
              }
            }
        })}
        />
      </div>
      }
      
      {
      inputs.currentLayer.type === 'conv2d'
      &&
      <div>
        <text>kernel</text>
        <input required
          placeholder="kernel size"
          onChange={(e) => setInputs({
            ...inputs,
            currentLayer : {
              type: inputs.currentLayer.type,
              props: {
                ...inputs.currentLayer.props,
                kernel : e.target.value,
              }
            }
        })}
        />
      </div>
      }

      {
      inputs.currentLayer.type === 'conv2d'
      &&
      <div>
        <text>stride</text>
        <input required
          placeholder="stride size"
          onChange={(e) => setInputs({
            ...inputs,
            currentLayer : {
              type: inputs.currentLayer.type,
              props: {
                ...inputs.currentLayer.props,
                stride : e.target.value,
              }
            }
        })}
        />
      </div>
      }

    <button onClick={addLayer}>
        Add Layer
    </button>

    <button onClick={showCode}>Show Code</button>

    </form>

    <div>{inputs.fileName}</div>

    <div>
      <button onClick={downloadZip}>Zip All</button>
      <button onClick={downloadFile}>Download python file</button>
    </div>

    <button
      disabled={inputs.fileName === "model.py"}
      onClick={() => setInputs({
        ...inputs,
        fileName : "model.py"
    })}
    >
     model.py
    </button>
    
    <button
      disabled={inputs.fileName === "train.py"}
      onClick={() => setInputs({
        ...inputs,
        fileName : "train.py"
    })}
    >
     train.py
    </button>

    <Editor
      width="80vw"
      height="60vh"
      path={files(inputs).name}
      defaultLanguage="python"
      defaultValue="You can do anything!"
      language={files(inputs).language}
      value={files(inputs).value}
      onMount={handleEditorDidMount}
    />
   </>
  );
}