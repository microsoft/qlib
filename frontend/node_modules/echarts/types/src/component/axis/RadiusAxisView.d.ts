import AxisView from './AxisView.js';
import { RadiusAxisModel } from '../../coord/polar/AxisModel.js';
import GlobalModel from '../../model/Global.js';
declare class RadiusAxisView extends AxisView {
    static readonly type = "radiusAxis";
    readonly type = "radiusAxis";
    axisPointerClass: string;
    private _axisGroup;
    render(radiusAxisModel: RadiusAxisModel, ecModel: GlobalModel): void;
}
export default RadiusAxisView;
