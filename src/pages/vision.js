import React, {useState, useEffect} from 'react'

export default function Component () {

  const [agreed, setAgreed] = useState(false)

  // This runs when the page is loaded.
  useEffect(() => {
    if (localStorage.getItem('agree')==="true") {
      setAgreed(true)
    }
    else{
      setAgreed(false)
    }
  }, [])

  const handleClick = () => {
    if (localStorage.getItem('agree')==="true") {
      localStorage.setItem('agree', 'false')
      setAgreed(false)
    }
    else{
      localStorage.setItem('agree', 'true')
      setAgreed(true)
    }
  }

  const AgreeButton = <button onClick={handleClick}>Click me to agree</button>

//  const AgreeButton = <button onClick={handleClick}>Click me to agree</button>

  return (
    <>
      <h1>Welcome to my page!</h1>
      {agreed
        ? <p>You agreed!</p>
        : <p>Not agreed!</p>
      }
      {AgreeButton}
    </>
  )
}