import React from "react";
import "./App.css";
import { useState, useRef, useCallback } from "react";

// import image1 image2 and image3
import image1 from './image1.jpeg';
import image2 from './image2.jpeg';
import image3 from './image3.jpeg';



function App() {
  const sliderRef = useRef(null);
  const [gradientPositions, setGradientPositions] = useState([25, 50, 75]); // Default positions for the gradient stops
  const [color1, setColor1] = useState('#FF0000');
  const [color2, setColor2] = useState('#00FF00');
  const [color3, setColor3] = useState('#0000FF');

   // Function to update the position of a gradient stop
   const updateGradientPosition = (index, position) => {
    setGradientPositions((prevPositions) => {
      const newPositions = [...prevPositions];
      newPositions[index] = position;
      return newPositions;
    });
  };

  const extractMainColor = (image) => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 1;
    canvas.height = 1;

    context.drawImage(image, 0, 0, 1, 1);
    let [r, g, b] = context.getImageData(0, 0, 1, 1).data;
    // Increase saturation by 10%
    r = Math.min(255, r * 1.4);
    g = Math.min(255, g * 1.5);
    b = Math.min(255, b * 1.8);
    return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
  };

  const blurAndExtractColors = () => {
    const img1 = new Image();
    img1.src = image1;
    const img2 = new Image();
    img2.src = image2;
    const img3 = new Image();
    img3.src = image3;

    img1.onload = () => {
      setColor1(extractMainColor(img1));
    };
    img2.onload = () => {
      setColor2(extractMainColor(img2));
    };
    img3.onload = () => {
      setColor3(extractMainColor(img3));
    };

  };

  // Call the function to extract colors
  blurAndExtractColors();


  // Generate a CSS gradient string based on the positions
  const gradient = `linear-gradient(to right, 
    ${color1} ${gradientPositions[0]}%, 
    ${color2} ${gradientPositions[1]}%, 
    ${color3} ${gradientPositions[2]}%)`;

  //   console.log(gradient);

  // make test gradient
  // const gradient = `linear-gradient(to right,
  // #FF0000 0%,
  // #00FF00 33%,
  // #FF00FF 100%)`;
  

  return (
    <div className="h-screen flex flex-col justify-center items-center bg-[#282828] space-y-8">
      {/* Black rectangle placeholder */}
      <div className="bg-[#1B1B1B] w-5/6 md:w-1/2 h-[400px] rounded-xl bg-[#1B1B1B] flex flex-row justify-center items-end space-x-8 p-12">

      {/* <SliderThumb sliderRef={sliderRef}/> */}

      <div className="relative w-full h-8" ref={sliderRef}>
      <div className="absolute inset-0 m-auto w-full h-12 rounded-lg border-[6px] border-[#444444]" 
      style={{background: gradient}}></div>
        <SliderThumb sliderRef={sliderRef} className=" hover:scale-x-150 active:scale-x-150 -mt-[2px]"/>
        {gradientPositions.map((position, index) => (

        <SliderImageThumb  key={index} index={index}  sliderRef={sliderRef} className="-mt-[60px] h-[45px] w-[45px]" updateColorPosition={updateGradientPosition}/>
        ))}


        </div>

      </div>

        {/* button that when pressed prints the location of the gradient */}
        <button 
        className="bg-[#1B1B1B] text-white rounded-lg p-2"
        onClick={() => console.log(gradientPositions.map(pos => (pos / 100).toFixed(3)))}>Print normalized gradient positions</button>

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
          className={`w-2 h-[35px] bg-white rounded-md absolute cursor-pointer transition-transform active:translate-y-[-10px] active:scale-150 hover:scale-105 ${className}`}
          style={{ left: `${position}%`, backgroundImage: `url(${index === 0 ? image1 : index === 1 ? image2 : image3})`, backgroundSize: 'cover' }}
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
          className={`w-4 h-[35px] bg-white rounded-md absolute cursor-pointer transition-transform ${className} `}
          style={{ left: `${position}%` }}
          onMouseDown={startDrag}
        >
          {/* Draggable thumb */}
        </div>
      </div>
  );
};

export default App;
