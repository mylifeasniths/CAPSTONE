const canvas = document.querySelector("canvas");
const clearBtn = document.querySelector(".clear-btn");
const saveBtn = document.querySelector(".save-btn");

const context = canvas.getContext("2d");
let currentSize = 10;
let isEraser = false;
let bucketColor = "#FFFFFF";
let currentColor = "#A51DAB";
let isMouseDown = false;
let drawnArray = [];

function resetCanvas() {
  context.beginPath();
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.rect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#FFFFFF";
  context.fill();
}

function getMousePosition(event) {
  const boundaries = canvas.getBoundingClientRect();
  return {
    x: event.clientX - boundaries.left,
    y: event.clientY - boundaries.top,
  };
}

function storeDrawn(x, y, size) {
  const line = { x, y, size };
  drawnArray.push(line);
}

// Mouse Down
canvas.addEventListener("mousedown", (event) => {
  isMouseDown = true;
  const currentPosition = getMousePosition(event);
  context.moveTo(currentPosition.x, currentPosition.y);
  context.beginPath();
  context.lineWidth = currentSize;
  context.lineCap = "round";
});

// Mouse Move
canvas.addEventListener("mousemove", (event) => {
  if (isMouseDown) {
    const currentPosition = getMousePosition(event);
    context.lineTo(currentPosition.x, currentPosition.y);
    context.stroke();
    // storeDrawn(currentPosition.x, currentPosition.y, currentSize);
  } else {
    // storeDrawn(undefined);
  }
});

// Mouse Up
canvas.addEventListener("mouseup", () => {
  isMouseDown = false;
});

clearBtn.addEventListener("click", resetCanvas);

saveBtn.addEventListener("click", async () => {
  const data = canvas.toDataURL("image/jpeg", 1);
  console.log(data);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: JSON.stringify({ data: data }),
    });
    const answer = await response.json();
    console.log(answer);
  } catch (error) {
    console.log(error);
  }
  resetCanvas();
});

resetCanvas();
