import AxisView from './AxisView.js';
import SingleAxisModel from '../../coord/single/AxisModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { Payload } from '../../util/types.js';
declare class SingleAxisView extends AxisView {
    static readonly type = "singleAxis";
    readonly type = "singleAxis";
    private _axisGroup;
    axisPointerClass: string;
    render(axisModel: SingleAxisModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    remove(): void;
}
export default SingleAxisView;
