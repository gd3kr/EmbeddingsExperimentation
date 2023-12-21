import React from "react";
import "./App.css";
import { useState, useRef, useCallback } from "react";
import axios from "axios";

// import image1 image2 and image3
import image1 from "./image5.jpeg";
import image2 from "./image2.jpeg";
import image3 from "./image3.jpeg";

function App() {

  const sliderRef = useRef(null);
  const [gradientPositions, setGradientPositions] = useState([25, 50, 75]); // Default positions for the gradient stops
  const [color1, setColor1] = useState("#FF0000");
  const [color2, setColor2] = useState("#00FF00");
  const [color3, setColor3] = useState("#0000FF");
  const [displayedImage, setDisplayedImage] = useState(image1);
  const [images, setImages] = useState([]); // Array of images to be displayed

  const convertImageToBase64 = (imgPath, callback) => {
    fetch(imgPath)
      .then(response => response.blob())
      .then(blob => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
          const base64data = reader.result;
          callback(base64data);
        };
      })
      .catch(error => console.error('Error:', error));
  };

  const storeLatents = (img, id) => {
    convertImageToBase64(img, (base64) => {
    //  remove the header from the base64 string
    base64 = base64.split(",")[1];
    
    axios.post("http://127.0.0.1:5000/store_latent", { image_b64: base64, id: id })
      .then(response => console.log(response.data))
      .catch(error => console.error('Error:', error));

    });
  };

  // Function to update the position of a gradient stop
  const updateGradientPosition = (index, position) => {
    setGradientPositions((prevPositions) => {
      const newPositions = [...prevPositions];
      newPositions[index] = position;
      return newPositions;
    });
  };

  const extractMainColor = (image) => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
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
    <div className="h-screen flex flex-col justify-center items-center bg-[#282828]">
      {/* Black rectangle placeholder */}
      <div className="bg-[#1B1B1B] w-5/6 m-[10px] p-12  rounded-xl  flex flex-col justify-around items-center gap-[100px]">
        <h1 className="text-4xl text-white font-bold">
          {/* Latent Surfing */}
        </h1>
        
        {/* create an image in the center which shows displayed Image */}
        <img
          src={`data:image/png;base64,${displayedImage}`}
          className="h-[250px] w-[250px] rounded-xl shadow-xl"
          alt="displayed image"
        />

        {/* create a slider to change the image */}

        {/* <SliderThumb sliderRef={sliderRef}/> */}

        <div className="relative w-full h-8" ref={sliderRef}>
          <div
            className="absolute inset-0 m-auto w-full h-12 rounded-lg border-[6px] border-[#444444]"
            style={{ background: gradient }}
          ></div>
          <SliderThumb
            sliderRef={sliderRef}
            className=" hover:scale-x-150 active:scale-x-150 -mt-[2px]"
            images={images}
            setImages={setImages}
            setDisplayedImage={setDisplayedImage}
          />
          {gradientPositions.map((position, index) => (
            <SliderImageThumb
              key={index}
              index={index}
              sliderRef={sliderRef}
              className="-mt-[60px] h-[45px] w-[45px]"
              updateColorPosition={updateGradientPosition}
            />
          ))}
        </div>
      </div>

      {/* button that when pressed prints the location of the gradient */}
      <button
        className="bg-[#1B1B1B] text-white rounded-lg p-2"
        onClick={async () =>{
          // const normalizedPositions = gradientPositions.map((pos) => (pos / 100).toFixed(3));
          // console.log(normalizedPositions);
          await storeLatents(image1, 1);
          await storeLatents(image2, 2);
          await storeLatents(image3, 3);
        }
        }
      >
        Compute Latents
      </button>
      <button
        className="bg-[#1B1B1B] text-white rounded-lg p-2"
        onClick={() =>{ 

          // delete all images
          setImages([]);

          const options = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'User-Agent': 'insomnia/8.4.5' },
            body: JSON.stringify({
                id_a: 1,
                id_b: 2,
                id_c: 3,
              num_images: 100,
              // positions: [0.1, 0.2, 0.8]
              positions: gradientPositions.map((pos) => parseFloat((pos / 100).toFixed(1)))
            })
          };

          console.log(options)
          // return;

          fetch('http://127.0.0.1:5000/pregenerate', options)
        .then(response => response.json())
        .then(response => {
          console.log(response)
          setImages(response.images)
          // images = response.images
          // Distribute images based on the position of the button

        })
        .catch(err => console.error(err));
        }
        }
      >
        Generate
      </button>

      {/* Progress bar/slider placeholder */}

      {/* Placeholder for additional content, if needed */}
      {/* <div className="w-5/6 md:w-1/2 h-10 bg-gray-300 rounded-lg"></div> */}
    </div>
  );
}

const SliderImageThumb = ({
  index,
  sliderRef,
  className,
  updateColorPosition,
}) => {
  const [position, setPosition] = useState((index + 1) * 30);

  const startDrag = (e) => {
    const slider = sliderRef.current;
    console.log(slider);

    const updatePosition = (e) => {
      if (!slider) return; // Ensure that slider is not null or undefined

      const rect = slider.getBoundingClientRect();
      const newPosition = e.clientX - rect.left;
      const endPosition = rect.width;

      if (newPosition >= 0 && newPosition <= endPosition-50) {
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
        style={{
          left: `${position}%`,
          backgroundImage: `url(${
            index === 0 ? image1 : index === 1 ? image2 : image3
          })`,
          backgroundSize: "cover",
        }}
        onMouseDown={startDrag}
      >
        {/* Draggable thumb */}
      </div>
    </div>
  );
};

const SliderThumb = ({ sliderRef, className, images, setImages, setDisplayedImage}) => {
  const [position, setPosition] = useState(0);

  const startDrag = (e) => {
    const slider = sliderRef.current;
    console.log(slider);

    const updatePosition = (e) => {
      if (!slider) return; // Ensure that slider is not null or undefined

      const rect = slider.getBoundingClientRect();
      const newPosition = e.clientX - rect.left;
      const endPosition = rect.width;

      if (newPosition >= 0 && newPosition <= endPosition) {
        const positionPercent = (newPosition / endPosition) * 100;
        setPosition(positionPercent);
        const imageIndex = Math.round(positionPercent * (images.length - 1) / 100);
        setDisplayedImage(images[imageIndex]);
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
