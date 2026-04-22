import Axis from '../Axis.js';
import Scale from '../../scale/Scale.js';
import { OptionAxisType } from '../axisCommonTypes.js';
import SingleAxisModel, { SingleAxisPosition } from './AxisModel.js';
import { LayoutOrient } from '../../util/types.js';
import Single from './Single.js';
interface SingleAxis {
    /**
     * Transform global coord to local coord,
     * i.e. let localCoord = axis.toLocalCoord(80);
     */
    toLocalCoord(coord: number): number;
    /**
     * Transform global coord to local coord,
     * i.e. let globalCoord = axis.toLocalCoord(40);
     */
    toGlobalCoord(coord: number): number;
}
declare class SingleAxis extends Axis {
    position: SingleAxisPosition;
    orient: LayoutOrient;
    coordinateSystem: Single;
    model: SingleAxisModel;
    constructor(dim: string, scale: Scale, coordExtent: [number, number], axisType?: OptionAxisType, position?: SingleAxisPosition);
    /**
     * Judge the orient of the axis.
     */
    isHorizontal(): boolean;
    pointToData(point: number[], clamp?: boolean): number;
}
export default SingleAxis;
