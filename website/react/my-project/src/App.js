import React from "react";
import "./App.css";
import { useState, useRef, useCallback } from "react";

function App() {
  const sliderRef = useRef(null);
  const [gradientPositions, setGradientPositions] = useState([25, 50, 75]); // Default positions for the gradient stops

   // Function to update the position of a gradient stop
   const updateGradientPosition = (index, position) => {
    setGradientPositions((prevPositions) => {
      const newPositions = [...prevPositions];
      newPositions[index] = position;
      return newPositions;
    });
  };

  // Generate a CSS gradient string based on the positions
  const gradient = `linear-gradient(to right, 
    #08D82F ${gradientPositions[0]}%, 
    #2B73FF ${gradientPositions[1]}%, 
    #D52BFF ${gradientPositions[2]}%)`;

  return (
    <div className="h-screen flex flex-col justify-center items-center bg-[#282828] space-y-8">
      {/* Black rectangle placeholder */}
      <div className="bg-[#1B1B1B] w-5/6 md:w-3/4 h-[400px] rounded-xl bg-[#1B1B1B] flex flex-row justify-center items-end space-x-8 p-12">

      {/* <SliderThumb sliderRef={sliderRef}/> */}



      <div className="relative w-full h-8" ref={sliderRef}>
      <div className="absolute inset-0 m-auto w-full h-12 bg-gradient-to-r from-green-400 via-blue-500 to-purple-600 rounded-lg border-2 border-gray-700" 
      style={{backgroundImage: gradient}}></div>
        <SliderThumb sliderRef={sliderRef} className=" hover:scale-x-150 active:scale-x-150 -mt-[2px]"/>
        {gradientPositions.map((position, index) => (

        <SliderImageThumb  key={index} index={index}  sliderRef={sliderRef} className="-mt-[60px] h-[45px] w-[45px]" updateColorPosition={updateGradientPosition}/>
        ))}


        </div>

      </div>

      {/* Progress bar/slider placeholder */}

      {/* Placeholder for additional content, if needed */}
      {/* <div className="w-5/6 md:w-1/2 h-10 bg-gray-300 rounded-lg"></div> */}
    </div>
  );
}

const SliderImageThumb = ({index, sliderRef, className, updateColorPosition}) => {
  const [position, setPosition] = useState((index+1)*30);


  const startDrag = (e) => {
    const slider = sliderRef.current;
    console.log(slider)

    const updatePosition = (e) => {
      if (!slider) return; // Ensure that slider is not null or undefined

      const rect = slider.getBoundingClientRect();
      const newPosition = e.clientX - rect.left;
      const endPosition = rect.width;

      if (newPosition >= 0 && newPosition <= endPosition) {
        const positionPercent = (newPosition / endPosition) * 100;
        setPosition(positionPercent);
        updateColorPosition(index, positionPercent);

      }
    };

    // Add mousemove and mouseup listeners
    document.addEventListener("mousemove", updatePosition);
    document.addEventListener("mouseup", () => {
      document.removeEventListener("mousemove", updatePosition);
    });
  };

  return (
    <div>
        <div
          className={`w-2 h-[35px] bg-white rounded-md absolute cursor-pointer transition-transform ${className}`}
          style={{ left: `${position}%` }}
          onMouseDown={startDrag}
        >
          {/* Draggable thumb */}
        </div>
      </div>
  );
};

const SliderThumb = ({sliderRef, className}) => {
  const [position, setPosition] = useState(0);


  const startDrag = (e) => {
    const slider = sliderRef.current;
    console.log(slider)

    const updatePosition = (e) => {
      if (!slider) return; // Ensure that slider is not null or undefined

      const rect = slider.getBoundingClientRect();
      const newPosition = e.clientX - rect.left;
      const endPosition = rect.width;

      if (newPosition >= 0 && newPosition <= endPosition) {
        setPosition((newPosition / endPosition) * 100);
      }
    };

    // Add mousemove and mouseup listeners
    document.addEventListener("mousemove", updatePosition);
    document.addEventListener("mouseup", () => {
      document.removeEventListener("mousemove", updatePosition);
    });
  };

  return (
    <div>
        <div
          className={`w-2 h-[35px] bg-white rounded-md absolute cursor-pointer transition-transform ${className}`}
          style={{ left: `${position}%` }}
          onMouseDown={startDrag}
        >
          {/* Draggable thumb */}
        </div>
      </div>
  );
};

export default App;
