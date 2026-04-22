import LegendView from './LegendView.js';
import { LegendSelectorButtonOption } from './LegendModel.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GlobalModel from '../../model/Global.js';
import ScrollableLegendModel, { ScrollableLegendOption } from './ScrollableLegendModel.js';
interface PageInfo {
    contentPosition: number[];
    pageCount: number;
    pageIndex: number;
    pagePrevDataIndex: number;
    pageNextDataIndex: number;
}
declare class ScrollableLegendView extends LegendView {
    static type: "legend.scroll";
    type: "legend.scroll";
    newlineDisabled: boolean;
    private _containerGroup;
    private _controllerGroup;
    private _currentIndex;
    private _showController;
    init(): void;
    /**
     * @override
     */
    resetInner(): void;
    /**
     * @override
     */
    renderInner(itemAlign: ScrollableLegendOption['align'], legendModel: ScrollableLegendModel, ecModel: GlobalModel, api: ExtensionAPI, selector: LegendSelectorButtonOption[], orient: ScrollableLegendOption['orient'], selectorPosition: ScrollableLegendOption['selectorPosition']): void;
    /**
     * @override
     */
    layoutInner(legendModel: ScrollableLegendModel, itemAlign: ScrollableLegendOption['align'], maxSize: {
        width: number;
        height: number;
    }, isFirstRender: boolean, selector: LegendSelectorButtonOption[], selectorPosition: ScrollableLegendOption['selectorPosition']): import("zrender/lib/core/BoundingRect").RectLike;
    _layoutContentAndController(legendModel: ScrollableLegendModel, isFirstRender: boolean, maxSize: {
        width: number;
        height: number;
    }, orientIdx: 0 | 1, wh: 'width' | 'height', hw: 'width' | 'height', yx: 'x' | 'y', xy: 'y' | 'x'): import("zrender/lib/core/BoundingRect").RectLike;
    _pageGo(to: 'pagePrevDataIndex' | 'pageNextDataIndex', legendModel: ScrollableLegendModel, api: ExtensionAPI): void;
    _updatePageInfoView(legendModel: ScrollableLegendModel, pageInfo: PageInfo): void;
    /**
     *  contentPosition: Array.<number>, null when data item not found.
     *  pageIndex: number, null when data item not found.
     *  pageCount: number, always be a number, can be 0.
     *  pagePrevDataIndex: number, null when no previous page.
     *  pageNextDataIndex: number, null when no next page.
     * }
     */
    _getPageInfo(legendModel: ScrollableLegendModel): PageInfo;
    _findTargetItemIndex(targetDataIndex: number): number;
}
export default ScrollableLegendView;
