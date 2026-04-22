import Model from '../model/Model.js';
import Axis from './Axis.js';
import { AxisBaseOption } from './axisCommonTypes.js';
import { CoordinateSystemHostModel } from './CoordinateSystem.js';
interface AxisModelCommonMixin<Opt extends AxisBaseOption> extends Pick<Model<Opt>, 'option'> {
    axis: Axis;
}
declare class AxisModelCommonMixin<Opt extends AxisBaseOption> {
    getNeedCrossZero(): boolean;
    /**
     * Should be implemented by each axis model if necessary.
     * @return coordinate system model
     */
    getCoordSysModel(): CoordinateSystemHostModel;
}
export { AxisModelCommonMixin };
