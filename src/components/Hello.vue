
<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>

    <h2>Enter image URL</h2>
    <input
      v-model="imageURLInput"
      spellcheck="false"
      @keyup.native.enter="onImageURLInputEnter"
    >
    <div class="error" v-if="imageLoadingError">Error loading URL</div>
    <div class="canvas-container">
      <canvas id="input-canvas" width="224" height="224"></canvas>
      <canvas id="output-canvas" width="224" height="224"></canvas>
    </div>
    <div v-if="imageLoading || modelRunning">Calculating</div>
  </div>
</template>

<script lang="babel">

const MODEL_FILEPATHS_DEV = {
  model: 'static/model/model.json',
  weights: 'static/model/model_weights.buf',
  metadata: 'static/model/model_metadata.json'
}
//const MODEL_FILEPATHS_PROD = {
//  model: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn.json',
//  weights: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn_weights.buf',
//  metadata: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn_metadata.json'
//}
const MODEL_CONFIG = { filepaths: MODEL_FILEPATHS_DEV, gpu: true }

const COLOR_CODES = [
  [128, 128, 128],
  [128, 0, 0],
  [192, 192, 128],
  [128, 64, 128],
  [0, 0, 192],
  [128, 128, 0],
  [192, 128, 128],
  [64, 64, 128],
  [64, 0, 128],
  [64, 64, 0],
  [0, 128, 192],
  [0, 0, 0]
]
const NAMES = ['sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree',
  'sign', 'fence', 'car', 'pedestrian', 'bicyclist', 'void']
import * as KerasJS from 'keras-js'
import loadImage from 'blueimp-load-image'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import * as utils from '../utils'

window.ndarray = ndarray
window.ops = ops
window.utils = utils

export default {
  name: 'hello',
  data () {
    return {
      model: new KerasJS.Model(MODEL_CONFIG), // eslint-disable-line
      modelLoading: true,
      imageURLInput: 'static/container.jpg',
      imageURLSelect: null,
      imageLoading: false,
      imageLoadingError: false,
      modelRunning: false,
      output: null,
      msg: 'Welcome to Your Vue.js App'
    }
  },
  computed: {
    loadingProgress: function () {
      return this.model.getLoadingProgress()
    }
  },
  mounted: function () {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(() => {
        this.loadImageToCanvas(this.imageURLInput)
      })
    })
  },
  methods: {
    onImageURLInputEnter: function(e) {
      this.loadImageToCanvas(e.target.value)
    },
    loadImageToCanvas: function(url) {
      if (!url) {
        this.clearAll()
        return
      }

      this.imageLoading = true
      loadImage(
        url,
        img => {
          if (img.type === 'error') {
            this.imageLoadingError = true
            this.imageLoading = false
          } else {
            // load image data onto input canvas
            const ctx = document.getElementById('input-canvas').getContext('2d')
            ctx.drawImage(img, 0, 0)
            this.imageLoadingError = false
            this.imageLoading = false
            this.modelRunning = true
            // model predict
            this.$nextTick(function() {
              setTimeout(() => {
                this.runModel()
              }, 200)
            })
          }
        },
        { maxWidth: 224, maxHeight: 224, cover: true, crop: true, canvas: true, crossOrigin: 'Anonymous' }
      )
    },
    runModel: function() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

//      // data processing
//      // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
//      // and https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py
      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))
      const inputData = { input_9: dataProcessedTensor.data }
      console.log("start")
      this.model.predict(inputData).then(outputData => {
        console.log("done")
        this.modelRunning = false
        window.outputData = outputData = outputData.activation_614
        let argmaxArray = new Uint8ClampedArray(width*height*4)
        let ii=0;
        for(let xy=0; xy<width*height; xy++) {
          let max = 0
          let maxValue = 0
          for(let i=0; i<12; i++) {
            if(maxValue<outputData[ii]) {
              max = i
            }
            ii+=1
          }
          argmaxArray[xy*4] = COLOR_CODES[max][0]
          argmaxArray[xy*4+1] = COLOR_CODES[max][1]
          argmaxArray[xy*4+2] = COLOR_CODES[max][2]
          argmaxArray[xy*4+3] = 255
        }
        console.log('ii', ii)
        window.output = this.output = new ImageData(argmaxArray, width, height)
        this.drawOutput()
      })
    },
    drawOutput: function() {
      const ctx = document.getElementById('output-canvas').getContext('2d')
      ctx.putImageData(this.output, 0, 0)
    },
    clearAll: function() {
      this.modelRunning = false
      this.imageURLInput = null
      this.imageLoading = false
      this.imageLoadingError = false
      this.output = null

      this.model.layersWithResults = []

      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    }
  }

}

</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1, h2 {
  font-weight: normal;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  display: inline-block;
  margin: 0 10px;
}

a {
  color: #42b983;
}
</style>
