import ComponentView from '../../view/Component.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GeoModel from '../../coord/geo/GeoModel.js';
import { Payload } from '../../util/types.js';
import Element from 'zrender/lib/Element.js';
declare class GeoView extends ComponentView {
    static type: "geo";
    readonly type: "geo";
    private _mapDraw;
    private _api;
    private _model;
    focusBlurEnabled: boolean;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(geoModel: GeoModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    private _handleRegionClick;
    updateSelectStatus(model: GeoModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    findHighDownDispatchers(name: string): Element[];
    dispose(): void;
}
export default GeoView;
