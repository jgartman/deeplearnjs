
import {getChannels, getVecChannels} from '../packing_util';

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class BatchToSpaceNDPackedProgram implements GPGPUProgram {
  variableNames = ['BatchTensor'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;
  outputTensorDtype: string;
  batchTensorStrides: number[];

  constructor(
      blockShape: number[], batchTensorStrides: number[],
      batchTensorShape: number[], crops: number[][], numBlockDims: number,
      internalOutputShape: number[]) {
    this.outputShape = internalOutputShape;
    const rank = internalOutputShape.length;

    const spaceTensorDtype = getCoordsDataType(rank);

    const stridesDtype = getCoordsDataType(batchTensorStrides.length);
    const stridesSnippet = this.getSnippetFromArray(batchTensorStrides);

    const blockShapeDtype = getCoordsDataType(blockShape.length);
    const blockShapeSnippet = this.getSnippetFromArray(blockShape);

    const batchTensorDtype = getCoordsDataType(batchTensorShape.length);
    const batchTensorInitSnippet =
        this.getSnippetFromArray(Array(batchTensorShape.length).fill(0));
    const batchTensorShapeSnippet = this.getSnippetFromArray(batchTensorShape);

    const outputOrder = getVecChannels('space_tensor_pos', rank);
    const channels = getChannels('batch_tensor_pos', rank);
    // const sourceCoords = getSourceCoords(rank, channels);
    const innerDims = channels.slice(-2);
    // const coords =
    //     rank <= 1 ? 'batch_tensor_pos' : `vec2(${innerDims.join(',')})`;
    const dims = `batch_tensor_pos[0], batch_tensor_pos[1],
      batch_tensor_pos[2], batch_tensor_pos[3]`;
    const getc = `getChannel(getBatchTensor(${dims}), vec2(${innerDims}))`;

    const nextColumn =
        `++${outputOrder[rank - 1]} < ${this.outputShape[rank - 1]}`;

    this.userCode = `
    void main(){
      ${spaceTensorDtype} space_tensor_pos = getOutputCoords();
      ${stridesDtype} batch_tensor_strides = ${stridesDtype}${stridesSnippet};
      ${blockShapeDtype} block_shape = ${blockShapeDtype}${blockShapeSnippet};
      ${batchTensorDtype} batch_tensor_pos = ${batchTensorDtype}
        ${batchTensorInitSnippet};
      ${batchTensorDtype} batch_tensor_shape = ${batchTensorDtype}${
        batchTensorShapeSnippet};

      int spatial_block_idx = 0;
      for(int i = ${blockShape.length}; i > 0; i--){
        int n = int(space_tensor_pos[i] / block_shape[i - 1]);
        spatial_block_idx += n * batch_tensor_strides[i];
      }

      int space_tensor_space_idx = 0;
      int strides = 1;
      for(int i = ${blockShape.length}; i > 0; i--){
        int n = imod(space_tensor_pos[i], block_shape[i - 1]);

        space_tensor_space_idx += n * strides;
        strides *= block_shape[i - 1];
      }

      batch_tensor_pos[0] = space_tensor_space_idx;

      for(int i = ${blockShape.length}; i > 0; i--){
        int dim = imod(spatial_block_idx, batch_tensor_shape[i]);
        batch_tensor_pos[i] = dim;
        spatial_block_idx /= batch_tensor_shape[i];
      }

      batch_tensor_pos[${batchTensorShape.length} - 1]
        = space_tensor_pos[${batchTensorShape.length} - 1];

      vec4 result = vec4(0.);
      result[0] = ${getc};
      if(${nextColumn}){
        if(batch_tensor_shape[3] > 1){
          batch_tensor_pos[3]++;
          result[1] = ${getc};
          batch_tensor_pos[3]--;
        }
        else{
          batch_tensor_pos[0]++;
          result[1] = ${getc};
        }
      }
      --${outputOrder[rank - 1]};
      if(++${outputOrder[rank - 2]} < ${this.outputShape[rank - 2]}) {
        batch_tensor_pos[0]++;
        result[2] = ${getc};
        if(${nextColumn}) {
          if(batch_tensor_shape[3] > 1){
            batch_tensor_pos[3]++;
          }else{
            batch_tensor_pos[0]++;
          }
          result[3] = ${getc};
        }
      }
      setOutput(result);
    }
    `;
  }

  getSnippetFromArray(array: number[]) {
    let result = `(`;
    for (let i = 0; i < array.length; i++) {
      result += String(array[i]);
      if (i === array.length - 1) {
        result += ')';
      } else {
        result += `, `;
      }
    }
    return result;
  }
}
