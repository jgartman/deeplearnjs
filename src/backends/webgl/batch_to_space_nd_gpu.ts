import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class BatchToSpaceNDProgram implements GPGPUProgram {
  variableNames = ['BatchTensor'];
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

    this.userCode = `
    void main(){
      ${spaceTensorDtype} space_tensor_pos = getOutputCoords();
      ${stridesDtype} batch_tensor_strides = ${stridesDtype}${stridesSnippet};
      ${blockShapeDtype} block_shape = ${blockShapeDtype}${blockShapeSnippet};
      ${batchTensorDtype} batch_tensor_pos = ${batchTensorDtype}
        ${batchTensorInitSnippet};
      ${batchTensorDtype} batch_tensor_shape = ${batchTensorDtype}${
        batchTensorShapeSnippet};
        ivec2 crop_start = ivec2(0, 1);

      int spatial_block_idx = 0;
      for(int i = ${blockShape.length}; i > 0; i--){
        int n = int(space_tensor_pos[i] / block_shape[i - 1]);
        spatial_block_idx += n * batch_tensor_strides[i] +
          crop_start[i - 1] / block_shape[i - 1];
      }

      int space_tensor_space_idx = 0;
      int strides = 1;
      for(int i = ${blockShape.length}; i > 0; i--){
        int n = imod(space_tensor_pos[i], block_shape[i - 1]);
        space_tensor_space_idx += n * strides +
          imod(crop_start[i - 1], block_shape[i - 1]);
        strides *= block_shape[i - 1];
      }

      batch_tensor_pos[0] = space_tensor_space_idx
        * ${this.outputShape[0]} + space_tensor_pos[0];

      for(int i = ${blockShape.length}; i > 0; i--){
        int dim = imod(spatial_block_idx, batch_tensor_shape[i]);
        batch_tensor_pos[i] = dim;
        spatial_block_idx /= batch_tensor_shape[i];
      }

      batch_tensor_pos[${batchTensorShape.length} - 1]
        = space_tensor_pos[${batchTensorShape.length} - 1];

       // setOutput(float(space_tensor_space_idx));
      setOutput(getBatchTensor(batch_tensor_pos[0], batch_tensor_pos[1],
         batch_tensor_pos[2], batch_tensor_pos[3]));
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
