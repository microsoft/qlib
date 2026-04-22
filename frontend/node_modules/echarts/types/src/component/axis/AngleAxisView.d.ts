import AxisView from './AxisView.js';
import { AngleAxisModel } from '../../coord/polar/AxisModel.js';
import GlobalModel from '../../model/Global.js';
declare class AngleAxisView extends AxisView {
    static readonly type = "angleAxis";
    readonly type = "angleAxis";
    axisPointerClass: string;
    render(angleAxisModel: AngleAxisModel, ecModel: GlobalModel): void;
}
export default AngleAxisView;
