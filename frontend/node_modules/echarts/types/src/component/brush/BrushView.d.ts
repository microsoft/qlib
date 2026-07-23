import BrushModel from './BrushModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { Payload } from '../../util/types.js';
import ComponentView from '../../view/Component.js';
declare class BrushView extends ComponentView {
    static type: string;
    readonly type: string;
    ecModel: GlobalModel;
    api: ExtensionAPI;
    model: BrushModel;
    private _brushController;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(brushModel: BrushModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    updateTransform(brushModel: BrushModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    updateVisual(brushModel: BrushModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    updateView(brushModel: BrushModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    private _updateController;
    dispose(): void;
    private _onBrush;
}
export default BrushView;
