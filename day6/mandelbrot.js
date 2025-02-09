const wasm = await WebAssembly.instantiateStreaming(fetch('mandelbrot.wasm'))
const {
  memory,
  generate_mandelbrot: generateMandelBrot,
  get_img_buf: getImgBuf,
} = wasm.instance.exports
const canvas = document.querySelector('canvas')
const ctx = canvas.getContext('2d')
const MAX_WIDTH = 3840
const MAX_HEIGHT = 2160

// canvas.width = Math.max(window.innerWidth, MAX_WIDTH)
// canvas.height = Math.max(window.innerHeight, MAX_HEIGHT)

canvas.width = 800
canvas.height = 800

const width = canvas.width
const height = canvas.height
const { left: canvasLeft, top: canvasTop } = canvas.getBoundingClientRect()

const imgBuf = getImgBuf()

let x0 = 0
let y0 = 0
let scaleFactor = 4 / canvas.width
const drag = { active: false, left: 0, top: 0 }

const render = () => {
  generateMandelBrot(width, height, x0, y0, scaleFactor)
  const newData = new Uint8ClampedArray(
    memory.buffer,
    imgBuf,
    4 * width * height,
  )
  const imgData = ctx.getImageData(0, 0, width, height)
  imgData.data.set(newData)
  ctx.putImageData(imgData, 0, 0)
}

canvas.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return

  drag.left = e.clientX - canvasLeft
  drag.top = e.clientY - canvasTop
  drag.active = true
  canvas.style.cursor = 'grabbing'
})

canvas.addEventListener('mouseup', (e) => {
  if (e.button !== 0) return

  drag.active = false
  canvas.style.cursor = 'default'
})

canvas.addEventListener('mouseout', (e) => {
  drag.active = false
  canvas.style.cursor = 'default'
})

canvas.addEventListener('mousemove', (e) => {
  if (!drag.active) return

  const x = e.clientX - canvasLeft
  const y = e.clientY - canvasTop

  const dx = x - drag.left
  const dy = y - drag.top

  x0 -= dx * scaleFactor
  y0 -= dy * scaleFactor

  render()

  // for next iter
  drag.left = x
  drag.top = y
})

canvas.addEventListener('wheel', (e) => {
  e.preventDefault() // prevent page scroll

  let x = e.clientX - canvasLeft
  let y = e.clientY - canvasTop

  const zoomCoeff = e.deltaY > 0 ? 0.1 : -0.1

  x = x0 + (x - width / 2) * scaleFactor
  y = y0 + (y - height / 2) * scaleFactor
  x0 -= (x - x0) * zoomCoeff
  y0 -= (y - y0) * zoomCoeff
  scaleFactor *= 1 + zoomCoeff

  render()
})

render()
