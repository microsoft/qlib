import TimelineView from './TimelineView.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import SliderTimelineModel from './SliderTimelineModel.js';
declare class SliderTimelineView extends TimelineView {
    static type: string;
    type: string;
    api: ExtensionAPI;
    model: SliderTimelineModel;
    ecModel: GlobalModel;
    private _axis;
    private _viewRect;
    private _timer;
    private _currentPointer;
    private _progressLine;
    private _mainGroup;
    private _labelGroup;
    private _tickSymbols;
    private _tickLabels;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    /**
     * @override
     */
    render(timelineModel: SliderTimelineModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    /**
     * @override
     */
    remove(): void;
    /**
     * @override
     */
    dispose(): void;
    private _layout;
    private _position;
    private _createAxis;
    private _createGroup;
    private _renderAxisLine;
    private _renderAxisTick;
    private _renderAxisLabel;
    private _renderControl;
    private _renderCurrentPointer;
    private _handlePlayClick;
    private _handlePointerDrag;
    private _handlePointerDragend;
    private _pointerChangeTimeline;
    private _doPlayStop;
    private _toAxisCoord;
    private _findNearestTick;
    private _clearTimer;
    private _changeTimeline;
    private _updateTicksStatus;
}
export default SliderTimelineView;
