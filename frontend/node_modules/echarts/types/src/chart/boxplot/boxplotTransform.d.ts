import { DataTransformOption, ExternalDataTransform } from '../../data/helper/transform.js';
import { PrepareBoxplotDataOpt } from './prepareBoxplotData.js';
export interface BoxplotTransformOption extends DataTransformOption {
    type: 'boxplot';
    config: PrepareBoxplotDataOpt;
}
export declare const boxplotTransform: ExternalDataTransform<BoxplotTransformOption>;
