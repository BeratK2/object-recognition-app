import { StatusBar } from 'expo-status-bar';
import { Dimensions, LogBox, Platform, StyleSheet, Text, View } from 'react-native';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import { Camera } from 'expo-camera';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import Canvas from 'react-native-canvas';


export default function App() {

  
  useEffect(() => {

    async function setup() {
      const {status} = await Camera.requestCameraPermissionsAsync();
      await tf.ready();
      setModel(await cocoSsd.load());
    }
    
    setup();
  
  }, []);
  

  const TensorCamera = cameraWithTensors(Camera);

  const {width, height} = Dimensions.get('window');

  LogBox.ignoreAllLogs(true);


  const [model, setModel] = useState<cocoSsd.ObjectDetection>();
  let context = useRef<CanvasRenderingContext2D>();
  let canvas = useRef<Canvas>();


  let textureDims = Platform.OS == 'ios' ? {height: 1920, width: 1080} : {height: 1200, width: 1600}

  const handleCameraStream = (images: any) => {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      if(!model || !nextImageTensor) throw new Error("No model or image tensor");
      model.detect(nextImageTensor)
      .then((prediction) => {
        //Draw rectangles
        drawRectangle(prediction, nextImageTensor)
      })
      .catch((error) => {
        console.log(error)
      })

      requestAnimationFrame(loop);

    };
    loop();
  }

  function drawRectangle(predictions: cocoSsd.DetectedObject[], nextImageTensor : any){
    if(!context.current || !canvas.current) return;
    
    //Match size of camera preview
    const scaleWidth = width / nextImageTensor.shape[1];
    const scaleHeight = height / nextImageTensor.shape[0];

    const flipHorizontal = Platform.OS == 'ios' ? false : true;

    //Clear previous prediction
    context.current.clearRect(0,0, width, height);

    for(const prediction of predictions){
      const[x,y,width, height] = prediction.bbox;

      const boundingBoxX = flipHorizontal ? canvas.current.width - x * scaleHeight - width * scaleWidth : x * scaleWidth;
      const boundingBoxY = y * scaleHeight;

      //draw rectangle
      context.current.strokeRect(
        boundingBoxX,
        boundingBoxY,
        width * scaleWidth,
        height * scaleHeight
      );

      //Draw the label
      context.current.strokeText(
        prediction.class,
        boundingBoxX - 5,
        boundingBoxY - 5
      )
    }
  }

  async function handleCanvas(can: Canvas){
    if(can){
      can.width = width,
      can.height = height
      const ctx : CanvasRenderingContext2D = can.getContext('2d')
      ctx.fillStyle = 'red'
      ctx.strokeStyle =  'red'
      ctx.lineWidth = 3;

      context.current = ctx;
      canvas.current = can;
    }
  }




  return (
    <View style={styles.container}>
      <TensorCamera
        style={styles.camera}
        type={Camera.Constants.Type.back}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={200}
        resizeWidth={152}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
        useCustomShadersToResize={false}
      />
      <Canvas style={styles.canvas} ref={handleCanvas}></Canvas>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    width: '100%',
    height: '100%'
  },
  canvas: {
    position: 'absolute',
    zIndex: 1000000,
    width: '100%',
    height: '100%'
  }
});
