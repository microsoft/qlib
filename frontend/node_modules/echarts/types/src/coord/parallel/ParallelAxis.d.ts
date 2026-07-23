import Axis from '../Axis.js';
import Scale from '../../scale/Scale.js';
import { DimensionName } from '../../util/types.js';
import { OptionAxisType } from '../axisCommonTypes.js';
import AxisModel from './AxisModel.js';
import Parallel from './Parallel.js';
declare class ParallelAxis extends Axis {
    readonly axisIndex: number;
    model: AxisModel;
    coordinateSystem: Parallel;
    constructor(dim: DimensionName, scale: Scale, coordExtent: [number, number], axisType: OptionAxisType, axisIndex: number);
    isHorizontal(): boolean;
}
export default ParallelAxis;
