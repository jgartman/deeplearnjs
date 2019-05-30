import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';
// import * as shader_util from './shader_compiler_util';

export class SpaceToBatchNDProgram implements GPGPUProgram {
  variableNames = ['batchTensor'];
  outputShape: number[];
  userCode: string;
  rank: number;

  argsBatchTensorBatch: number;
  argsBlockShape: Int32Array;
  argsBatchTensorSpatialShape: Int32Array;
  argsCropsStart: Int32Array;
  numBlockDims: number;

  constructor(
      argsBatchTensorBatch: number, argsBlockShape: Int32Array,
      argsSpaceTensorShape: Int32Array, argsBatchTensorSpatialShape: Int32Array,
      argsCropStart: Int32Array, numBlockDims: number) {
    // this.outputShape = Array.prototype.slice.call(argsSpaceTensorShape);
    this.outputShape = [1, 2, 2, 1];
    this.rank = this.outputShape.length;
    const dtype = getCoordsDataType(this.rank);
    const spatialShapeDtype =
        getCoordsDataType(argsBatchTensorSpatialShape.length);

    this.argsBatchTensorBatch = 4;
    this.argsBlockShape = new Int32Array([2, 2]);
    this.argsBatchTensorSpatialShape = new Int32Array([1, 1]);
    this.argsCropsStart = argsCropStart = new Int32Array([0, 0]);
    this.numBlockDims = numBlockDims;

    // const batchTensorSpatialShapeSnippet =
    //     this.getBatchTensorSpatialShapeSnippet();

    this.userCode = `
    void main(){
      ${dtype} space_tensor_pos = getOutputCoords();
      ${spatialShapeDtype} spatial_shape = ${spatialShapeDtype}(${
        argsBatchTensorSpatialShape});

      int batch_tensor_idx = 0;

      // hardcode stuff for now
      ivec2 block_shape = ivec2(2, 2);
      ivec4 batch_tensor_shape = ivec4(4, 1, 1, 1);
      ivec2 batch_tensor_spatial_shape = ivec2(1, 1);
      int batch_tensor_batch = 4;

      int block_idx = 0;
      for(int i = 0 ; i < ${this.argsBatchTensorSpatialShape.length}; i++){
        block_idx += space_tensor_pos[i + 1] / spatial_shape[i];
      }

      int batch_tensor_batch_idx = 0;
      int batch_tensor_strides = batch_tensor_shape[${numBlockDims} + 1];
      for(int i = 0; i < ${this.argsBatchTensorSpatialShape.length}; i++){
        // this is using batch tensor spatial shape
        int n = int(pow(float(block_shape[i]), float(1 - i))) *
          space_tensor_pos[i + 1];
        int d = int(pow(float(block_shape[i]),float(2 - i)));
        batch_tensor_batch_idx += imod(n, d);
      }

      for(int i = 0; i < ${numBlockDims}; i++){
        int offset = block_idx;
        if(i > 0){
          offset = imod(offset, 4);
        }

        // need to use crops
        int batch_tensor_pos
          = space_tensor_pos[i + 1] * block_shape[i] + offset;
        batch_tensor_idx += batch_tensor_strides * batch_tensor_pos;
        batch_tensor_strides *= batch_tensor_spatial_shape[i];

        if(i == 0){
          batch_tensor_idx += batch_tensor_strides * batch_tensor_batch_idx;

        block_idx /= batch_tensor_batch;
        }
      }
      setOutput(float(batch_tensor_batch_idx));
    }
    `;
  }

  getBatchTensorSpatialShapeSnippet() {
    const result = [];
    for (let i = 0; i < this.argsBatchTensorSpatialShape.length; i++) {
      result.push(this.argsBatchTensorSpatialShape[i]);
    }
    return `[` + result.join() + `]`;
  }
}
