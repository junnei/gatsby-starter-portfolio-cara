import React, {useRef, useState } from 'react'

import Editor from "@monaco-editor/react";
import files from "../components/files";

export default function Component () {

  const editorRef = useRef(null);

  function handleEditorDidMount(editor, monaco) {
    editorRef.current = editor; 
  }
  
  function showValue() {
    alert(editorRef.current.getValue());
  }

  const [inputs, setInputs] = useState({
    fileName: "model.py",
    imageSizeX: "244",
    imageSizeY: "244"
  })

  return (
   <>
     <form>
      <text>image size(x)</text>
      <input
        placeholder="x"
        value={inputs.imageSizeX}
        onChange={(e) => setInputs({
          ...inputs,
          imageSizeX : e.target.value
      })}
      />
      <text>image size(y)</text>
      <input
        placeholder="y"
        value={inputs.imageSizeY}
        onChange={(e) => setInputs({
          ...inputs,
          imageSizeY : e.target.value
      })}
      />

      <div>{inputs.fileName}</div>
    </form>
    <button onClick={showValue}>Show value</button>

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

    <button
      disabled={inputs.fileName === "index.html"}
      onClick={() => setInputs({
        ...inputs,
        fileName : "index.html"
    })}
    >
     index.html
    </button>

    <Editor
      width="60vw"
      height="40vh"
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