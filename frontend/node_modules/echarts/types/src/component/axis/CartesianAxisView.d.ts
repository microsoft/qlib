import AxisView from './AxisView.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import CartesianAxisModel from '../../coord/cartesian/AxisModel.js';
import { Payload } from '../../util/types.js';
declare class CartesianAxisView extends AxisView {
    static type: string;
    type: string;
    axisPointerClass: string;
    private _axisGroup;
    /**
     * @override
     */
    render(axisModel: CartesianAxisModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    remove(): void;
}
export declare class CartesianXAxisView extends CartesianAxisView {
    static type: string;
    type: string;
}
export declare class CartesianYAxisView extends CartesianAxisView {
    static type: string;
    type: string;
}
export default CartesianAxisView;
