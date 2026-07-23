import * as graphic from '../../util/graphic.js';
import ComponentView from '../../view/Component.js';
import LegendModel, { LegendOption, LegendSelectorButtonOption } from './LegendModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { ZRRectLike } from '../../util/types.js';
declare class LegendView extends ComponentView {
    static type: string;
    type: string;
    newlineDisabled: boolean;
    private _contentGroup;
    private _backgroundEl;
    private _selectorGroup;
    /**
     * If first rendering, `contentGroup.position` is [0, 0], which
     * does not make sense and may cause unexpected animation if adopted.
     */
    private _isFirstRender;
    init(): void;
    /**
     * @protected
     */
    getContentGroup(): graphic.Group;
    /**
     * @protected
     */
    getSelectorGroup(): graphic.Group;
    /**
     * @override
     */
    render(legendModel: LegendModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    protected resetInner(): void;
    protected renderInner(itemAlign: LegendOption['align'], legendModel: LegendModel, ecModel: GlobalModel, api: ExtensionAPI, selector: LegendSelectorButtonOption[], orient: LegendOption['orient'], selectorPosition: LegendOption['selectorPosition']): void;
    private _createSelector;
    private _createItem;
    protected layoutInner(legendModel: LegendModel, itemAlign: LegendOption['align'], maxSize: {
        width: number;
        height: number;
    }, isFirstRender: boolean, selector: LegendOption['selector'], selectorPosition: LegendOption['selectorPosition']): ZRRectLike;
    /**
     * @protected
     */
    remove(): void;
}
export default LegendView;
