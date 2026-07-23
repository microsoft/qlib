import Element from 'zrender/lib/Element.js';
import GlobalModel from '../../model/Global.js';
import ComponentView from '../../view/Component.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { GraphicComponentModel, GraphicComponentElementOption } from './GraphicModel.js';
export declare const inner: (hostObj: Element<import("zrender/lib/Element").ElementProps>) => {
    width: number;
    height: number;
    isNew: boolean;
    id: string;
    type: string;
    option: GraphicComponentElementOption;
};
export declare class GraphicComponentView extends ComponentView {
    static type: string;
    type: string;
    private _elMap;
    private _lastGraphicModel;
    init(): void;
    render(graphicModel: GraphicComponentModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    /**
     * Update graphic elements.
     */
    private _updateElements;
    /**
     * Locate graphic elements.
     */
    private _relocate;
    /**
     * Clear all elements.
     */
    private _clear;
    dispose(): void;
}
