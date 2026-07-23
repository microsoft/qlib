import ComponentView from '../../view/Component.js';
import DataZoomModel from './DataZoomModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
declare class DataZoomView extends ComponentView {
    static type: string;
    type: string;
    dataZoomModel: DataZoomModel;
    ecModel: GlobalModel;
    api: ExtensionAPI;
    render(dataZoomModel: DataZoomModel, ecModel: GlobalModel, api: ExtensionAPI, payload: any): void;
}
export default DataZoomView;
