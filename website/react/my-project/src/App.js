import React from "react";
import "./App.css";
import { useState, useRef, useCallback } from "react";

function App() {
  return (
    <div className="h-screen flex flex-col justify-center items-center bg-[#282828] space-y-8">
      {/* Black rectangle placeholder */}
      <div className="bg-black w-5/6 md:w-1/2 h-96 rounded-xl bg-[#1B1B1B] flex flex-row justify-center items-end space-x-8 p-12">
        <SliderThumb />
      </div>

      {/* Progress bar/slider placeholder */}

      {/* Placeholder for additional content, if needed */}
      {/* <div className="w-5/6 md:w-1/2 h-10 bg-gray-300 rounded-lg"></div> */}
    </div>
  );
}

const SliderThumb = () => {
  const [position, setPosition] = useState(0);
  const sliderRef = useRef(null);

  const startDrag = (e) => {
    const slider = sliderRef.current;
    const updatePosition = (e) => {
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
      <div className="relative w-full h-8" ref={sliderRef}>
        <div className="absolute inset-0 m-auto w-full h-8 bg-gradient-to-r from-green-400 via-blue-500 to-purple-600 rounded-lg border-4 border-gray-700"></div>
        <div
          className="w-6 h-8 bg-white rounded-md absolute -mt-[5px] cursor-pointer"
          style={{ left: `${position}%` }}
          onMouseDown={startDrag}
        >
          {/* Draggable thumb */}
        </div>
      </div>
  );
};

export default App;
