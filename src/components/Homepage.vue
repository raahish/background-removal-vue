
<template>
  <div id="content" class="container">
    <div class="row">
      <div class="col-md-push-2 col-md-8 col-xs-12">

      </div>
    </div>
    <div class="text-center loading-progress" v-if="modelLoading">
      Loading Keras model...{{ loadingProgress }}%
    </div>

    <droply
      :class="{hidden: modelLoading}"
      ref="dropzone"
      id="dropzone"
      url="#"
      :dropzoneOptions="{resizeWidth: 448, resize: resize, parallelUploads: 1}"
      :parallelUploads="1"></droply>

    <div class="error" v-if="imageLoadingError">Error loading URL</div>
    <div class="results">
      <img :src="src" v-for="src of output"/>
    </div>
  </div>
</template>


<script lang="babel">

const MODEL_FILEPATHS_DEV = {
  model: 'static/model/tiramisu_2_classes.json',
  weights: 'static/model/tiramisu_2_classes_weights.buf',
  metadata: 'static/model/tiramisu_2_classes_metadata.json'
}
const BACKGROUND_CLASS = 0
const MODEL_CONFIG = { filepaths: MODEL_FILEPATHS_DEV, gpu: true, layerCallPauses: true }

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
//const NAMES = ['sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree',
//  'sign', 'fence', 'car', 'pedestrian', 'bicyclist', 'void']
import * as KerasJS from 'keras-js'
import _ from 'lodash'
import loadImage from 'blueimp-load-image'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import EXIF from 'exif-js'
import Droply from './Droply'
import Dropzone from 'dropzone'

window.EXIF = EXIF // Dropzone needs this as a global

export default {
  name: 'homepage',
  components: {
    Droply
  },
  data () {
    return {
      model: new KerasJS.Model(MODEL_CONFIG), // eslint-disable-line
      modelLoading: true,
      imageURLSelect: null,
      imageLoading: false,
      imageLoadingError: false,
      modelRunning: false,
      output: [],
      files: [],
      intervals: [],
      showIntermediates: false,
    }
  },
  computed: {
    loadingProgress: function () {
      return this.model.getLoadingProgress()
    },
  },
  mounted: function () {
    // https://stackoverflow.com/questions/33710825/getting-file-contents-when-using-dropzonejs
//    this.$refs.dropzone.dropzone.accept = this.localAcceptHandler
    this.$refs.dropzone.dropzone.uploadFiles = this.uploadFiles
    this.model.ready().then(() => {
      this.modelLoading = false
    })
  },
  methods: {
    // Helper function that uses HTML5 FileReader API in a form of a Promise
    readLocalFile: function(file) {
      return new Promise((resolve,reject) => {
        const reader = new window.FileReader();
        reader.onload = () => resolve(reader.result)
        reader.onerror = () => reject(reader.result)
        // run the reader
        reader.readAsDataURL(file);
      });
    },

    // Helper for runModel()
    loadNextImageToCanvas: function() {
      let file = this.files.pop()
      this.runModel(file)
    },

    // Override Dropzone.defaultOptions.resize method to allow 'squeeze' mode
    resize: function(file, width, height, resizeMethod) {
      if(resizeMethod=='contain' || resizeMethod=='crop') {
        return Dropzone.prototype.defaultOptions.resize(file, width, height, resizeMethod)
      }
      return {
        srcX: 0,
        srcY: 0,
        srcWidth: file.width,
        srcHeight: file.height,
        trgWidth: width,
        trgHeight: height,
      };
    },

    // Called by Dropzone when a new file is added
    uploadFiles: function(files) {
      files.forEach(file => {
        this.$refs.dropzone.dropzone.createThumbnail(file, 224, 224, 'squeeze', true, dataUrl => {
          console.log('original file WxH', file.width, file.height)
          const img =  this.$refs.inputImg || new Image
          img.onload = () => this.runModel(img, file)
          img.src=dataUrl
        })
      })
    },

    beforeDestroy: function() {
      this.intervals.forEach(i => clearInterval(i))
    },

    // Helper function to set progress bar
    updateDropzoneFileProgress: function(file, progress) {
      const self = this.$refs.dropzone.dropzone
      self.emit('uploadprogress', file, progress)
      if(progress == 100) {
        file.status = Dropzone.SUCCESS;
        self.emit("success", file, 'success', null);
        self.emit("complete", file);
        self.processQueue();
      }
    },

    file2ImageData: function(file) {
      return this.readLocalFile(file)
        .then(result => new Promise((resolve, reject) => {
          const img2 = this.$refs.originalImg || new Image
          img2.onload = () => resolve(img2)
          img2.onerror = reject
          img2.src = result
        })).then(img2 => {
          const canvas = document.createElement('canvas')
          canvas.width = img2.width
          canvas.height = img2.height
          const ctx = canvas.getContext('2d')
          ctx.drawImage(img2, 0, 0, img2.width, img2.height)
          return ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)

        })
    },

    runModel: function(img, file) {
      const canvas = this.$refs.inputCanvas || document.createElement('canvas')
      canvas.width = 224
      canvas.height = 224
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, img.width, img.height)
      const { data, width, height } = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)

//      // data processing
//      // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
//      // and https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py
      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))
      console.log('inputTensors', this.model.inputTensors)
      const inputKey = Object.keys(this.model.inputTensors)[0]
      const inputData = window.inputData = { [inputKey]: dataProcessedTensor.data }
      const n_layers = Object.keys(this.model.modelDAG).length
      console.log("start", n_layers)
      const intervalId = setInterval(() => {
        console.log(this.model.layersWithResults.length)
        this.updateDropzoneFileProgress(file, this.model.layersWithResults.length/n_layers*90)
      }, 1000)
      this.intervals.push(intervalId)
      return this.model.predict(inputData).then(outputData => {
        console.log("done")
        clearInterval(intervalId)
        this.intervals.splice(this.intervals.indexOf(intervalId))
        this.updateDropzoneFileProgress(file, 100)

        // Process output into Alpha channel
        let outputLayerName = _.values(this.model.modelDAG).filter(node => !node.outbound.length)[0].name
        outputData = outputData = outputData[outputLayerName]
        console.log('outputLayerName', outputLayerName)
        let n_activations = outputData.length / width / height
        console.log('n_activations', n_activations)
        const outputImage = new Uint8ClampedArray(width*height*4) // 4 channels RGBA
        let ii=0;
        for(let xy=0; xy<width*height; xy++) {
          let max = 0
          let maxValue = 0
          for(let i=0; i<n_activations; i++) {
            if(maxValue<outputData[ii]) {
              max = i
              maxValue = outputData[ii]
            }
            ii+=1
          }
//          outputImage[xy*4] = COLOR_CODES[max][0]
//          outputImage[xy*4+1] = COLOR_CODES[max][1]
//          outputImage[xy*4+2] = COLOR_CODES[max][2]
//          outputImage[xy*4+3] = 255

          outputImage[xy*4] = data[xy*4]
          outputImage[xy*4+1] = data[xy*4+1]
          outputImage[xy*4+2] = data[xy*4+2]
          outputImage[xy*4+3] = max!=BACKGROUND_CLASS ? maxValue*255 : 0
        }

        this.drawOutput(new ImageData(outputImage, width, height), file)
        if(this.files.length) {
          this.loadNextImageToCanvas()
        } else {
          this.modelRunning = false
        }
        console.log('done displaying')
      })
    },

    resizeOutput: function(imageData, file) {
      const canvas = document.createElement('canvas')
      canvas.width = imageData.width // 224
      canvas.height = imageData.height // 224
      const ctx = canvas.getContext('2d')
      ctx.putImageData(imageData, 0, 0)

      const resizedCanvas = this.$refs.resizedCanvas || document.createElement('canvas')
      resizedCanvas.width = file.width // original picture width
      resizedCanvas.height = file.height  // original picture height
      const resizedContext = resizedCanvas.getContext('2d')
      resizedContext.scale(file.width / imageData.width, file.height / imageData.height)
      resizedContext.drawImage(canvas, 0, 0)
      return resizedCanvas
    },

    drawOutput: function(imageData, file) {
      // Take the ALPHAs only from resizedImageData and combine with the original picture
      this.file2ImageData(file).then((originalImageData) => {
        const { data, width, height } = originalImageData

        const resizedCanvas = this.resizeOutput(imageData, file)
        const resizedContext = resizedCanvas.getContext('2d')
        const {data: resizedData} = resizedContext.getImageData(0, 0, file.width, file.height)

        let len = data.length
        for(let i=3; i<len; i+=4) {
          data[i] = resizedData[i]
        }
        resizedContext.putImageData(new ImageData(data, width, height), 0, 0)
        this.output.unshift(resizedCanvas.toDataURL('image/png'))
      })
    },
  }

}

</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
@import url('~dropzone/dist/dropzone.css');
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

.alert-danger {
  color: #fff;
  background-color: #ff3f49;
  margin: 30px 0;
}

.results img {
  max-width: 100%;
  max-height: 500px;
  display: block;
  margin: 30px auto;
}

.dropzone {
  border: 2px dashed #0087F7;
  border-radius: 5px;
  background: white;
}

.dropzone .dz-message {
  font-weight: 400;
}

.dropzone .dz-message .note {
  font-size: 0.8em;
  font-weight: 200;
  display: block;
  margin-top: 1.4rem;
}


</style>
