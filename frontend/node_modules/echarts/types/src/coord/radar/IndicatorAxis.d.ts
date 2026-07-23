import Axis from '../Axis.js';
import Scale from '../../scale/Scale.js';
import { OptionAxisType } from '../axisCommonTypes.js';
import { AxisBaseModel } from '../AxisBaseModel.js';
import { InnerIndicatorAxisOption } from './RadarModel.js';
declare class IndicatorAxis extends Axis {
    type: OptionAxisType;
    angle: number;
    name: string;
    model: AxisBaseModel<InnerIndicatorAxisOption>;
    constructor(dim: string, scale: Scale, radiusExtent?: [number, number]);
}
export default IndicatorAxis;
