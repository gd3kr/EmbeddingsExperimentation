import React from "react";
import "./App.css";
import { useState, useRef, useCallback } from "react";
import axios from "axios";

// import image1 image2 and image3
import image1 from "./image2.jpeg";
import image2 from "./image11.jpeg";
import image3 from "./image20.jpeg";
import waves from "./waves.png";

// load images from testImages.json

function App() {
  // const preloadedImages = require("./testimages.json").images;

  const sliderRef = useRef(null);
  const [gradientPositions, setGradientPositions] = useState([25, 50, 75]); // Default positions for the gradient stops
  const [color1, setColor1] = useState("#FF0000");
  const [color2, setColor2] = useState("#00FF00");
  const [color3, setColor3] = useState("#0000FF");
  const [images, setImages] = useState([]);
  const [displayedImage, setDisplayedImage] = useState(images[0]);


  const convertImageToBase64 = (imgPath, callback) => {
    fetch(imgPath)
      .then((response) => response.blob())
      .then((blob) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
          const base64data = reader.result;
          callback(base64data);
        };
      })
      .catch((error) => console.error("Error:", error));
  };

  const storeLatents = (img, id) => {
    convertImageToBase64(img, (base64) => {
      //  remove the header from the base64 string
      base64 = base64.split(",")[1];

      axios
        .post("http://127.0.0.1:5000/store_latent", {
          image_b64: base64,
          id: id,
        })
        .then((response) => console.log(response.data))
        .catch((error) => console.error("Error:", error));
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
    <div className="h-screen flex flex-col justify-center items-center bg-white">
      {/* Black rectangle placeholder */}


      <div className="w-[800px] ">
        <h1 className="ml-[10px] subheadline"> embeddings/</h1>
        <h1 className="ml-[10px] text-3xl headline"> Latent Surfing </h1>
        <div className="flex justify-center items-stretch m-2 w-full gap-2">
          <div className=" w-[400px] bg-[#F7F7F7]  p-12  rounded-xl  flex flex-col justify-around items-center gap-[100px]  border border-[#D2D2D2]">

            {/* three images taken as input in one column */}
            <div className="flex flex-col justify-around items-center gap-5">
              <h1 className="subheadline self-start">inputs</h1>
              <img
                src={image1}
                className=" rounded-xl shadow-xl w-full h-full object-cover"
                alt="image1"
              />
              <img
                src={image2}
                className="rounded-xl shadow-xl w-full h-full object-cover"
                alt="image2"
              />
              <img
                src={image3}
                className="rounded-xl shadow-xl w-full h-full object-cover"
                alt="image3"
              />
            </div>
          </div>


          <div className="w-[800px] bg-[#F7F7F7] p-12  rounded-xl  flex flex-col justify-center gap-[20px] items-center  border border-[#D2D2D2]">
            {/* create an image in the center which shows displayed Image */}
            <div
              className="absolute"
              style={{
                borderRadius: '26px',
                opacity: 0.8,
                background: displayedImage?.includes("data:image/jpeg;base64,")
                  ? `url(${displayedImage}) lightgray 50% / cover no-repeat`
                  : `url(data:image/jpeg;base64,${displayedImage}) lightgray 50% / cover no-repeat`,
                mixBlendMode: 'hard-light',
                filter: 'blur(52.599998474121094px)',
                width: '400px', // slightly larger than the front image
                height: '400px',
                bottom: '35%',
              }}
            />
            <img
              src={displayedImage?.includes("data:image/jpeg;base64,") ? displayedImage : `data:image/jpeg;base64,${displayedImage}`}
              className="h-[400px] w-[400px] rounded-xl shadow-xl relative"
              alt="displayed image"
            />

            <div className="relative w-full h-8 mt-[100px]" ref={sliderRef}>
              <div
                className="absolute inset-0 m-auto w-full h-12 rounded-lg"
                style={{ background: gradient }}
              ></div>
              <SliderThumb
                sliderRef={sliderRef}
                className=" active:scale-105 -mt-[2px]"
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
                  setImage={() => {
                    setDisplayedImage(index === 0 ? image1 : index === 1 ? image2 : image3)
                  }}
                />
              ))}
            </div>
            <div
              className="flex  w-full space-x-2  mx-2 items-stretch content-stretch justify-center"
            >

              <button
                className="bg-[#F7F7F7] border border-[#D2D2D2] text-black rounded-lg p-2 nanum w-full"
                onClick={async () => {
                  // const normalizedPositions = gradientPositions.map((pos) => (pos / 100).toFixed(3));
                  // console.log(normalizedPositions);
                  await storeLatents(image1, 1);
                  await storeLatents(image2, 2);
                  await storeLatents(image3, 3);
                }}
              >
                Compute Latents
              </button>
              <button
                className="bg-[#F7F7F7] border border-[#D2D2D2] text-black rounded-lg p-2 nanum w-full"
                onClick={() => {
                  // delete all images
                  setImages([]);

                  const options = {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                      "User-Agent": "insomnia/8.4.5",
                    },
                    body: JSON.stringify({
                      id_a: 1,
                      id_b: 2,
                      id_c: 3,
                      num_images: 300,
                      // positions: [0.1, 0.2, 0.8]
                      positions: gradientPositions.map((pos) =>
                        parseFloat((pos / 100).toFixed(1))
                      ),
                    }),
                  };

                  console.log(options);
                  // return;

                  fetch("http://127.0.0.1:5000/pregenerate", options)
                    .then((response) => response.json())
                    .then((response) => {
                      console.log(response);
                      setImages(response.images);
                      // images = response.images
                      // Distribute images based on the position of the button
                    })
                    .catch((err) => console.error(err));
                }}
              >
                Generate
              </button>
            </div>
          </div>
        </div>

      </div>

      {/* button that when pressed prints the location of the gradient */}


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
  setImage,
}) => {
  const [position, setPosition] = useState(index === 0 ? 0 : index === 1 ? 50 : 100)

  const startDrag = (e) => {
    const slider = sliderRef.current
    setImage()

    // set the current image to the one that is being dragged

    const updatePosition = (e) => {
      if (!slider) return; // Ensure that slider is not null or undefined

      const rect = slider.getBoundingClientRect();
      const newPosition = e.clientX - rect.left;
      const endPosition = rect.width;

      if (newPosition >= 3 && newPosition <= endPosition) {
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
          left: `${index == 0 ? position : index == 1 ? position - 5 : position - 10}%`,
          backgroundImage: `url(${index === 0 ? image1 : index === 1 ? image2 : image3
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

const SliderThumb = ({
  sliderRef,
  className,
  images,
  setImages,
  setDisplayedImage,
}) => {
  const [position, setPosition] = useState(1);

  const startDrag = (e) => {
    const slider = sliderRef.current;
    console.log(slider);

    const updatePosition = (e) => {
      if (!slider) return; // Ensure that slider is not null or undefined

      const rect = slider.getBoundingClientRect();
      const newPosition = e.clientX - rect.left;
      const endPosition = rect.width;

      if (newPosition >= 5 && newPosition <= endPosition) {
        const positionPercent = (newPosition / (endPosition)) * 100;
        setPosition(Math.max(positionPercent-10, 0));
        const imageIndex = Math.round(
          (positionPercent * (images.length - 1)) / 100
        );
        if (images[imageIndex] != null) {
          setDisplayedImage(images[imageIndex]);
        }
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
        className={`w-4 h-[35px] w-[35px] bg-white rounded-md absolute cursor-pointer transition-transform ${className} flex justify-center items-center`}
        style={{ left: `${position}%`, userSelect: 'none' }} // Added userSelect property here
        onMouseDown={startDrag}
      >
        {/* Draggable thumb */}
        <div className="flex justify-center space-x-1">
          <h1>🏄</h1>
        </div>
      </div>
    </div>
  );

};

export default App;
