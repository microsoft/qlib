import Axis from '../Axis.js';
import Scale from '../../scale/Scale.js';
import Polar from './Polar.js';
import { RadiusAxisModel } from './AxisModel.js';
interface RadiusAxis {
    dataToRadius: Axis['dataToCoord'];
    radiusToData: Axis['coordToData'];
}
declare class RadiusAxis extends Axis {
    polar: Polar;
    model: RadiusAxisModel;
    constructor(scale?: Scale, radiusExtent?: [number, number]);
    pointToData(point: number[], clamp?: boolean): number;
}
export default RadiusAxis;
