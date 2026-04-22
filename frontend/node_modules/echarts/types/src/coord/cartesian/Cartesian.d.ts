import { DimensionName } from '../../util/types.js';
import Axis from '../Axis.js';
declare class Cartesian<AxisT extends Axis> {
    readonly type: string;
    readonly name: string;
    private _dimList;
    private _axes;
    constructor(name: string);
    getAxis(dim: DimensionName): AxisT;
    getAxes(): AxisT[];
    getAxesByScale(scaleType: string): AxisT[];
    addAxis(axis: AxisT): void;
}
export default Cartesian;
