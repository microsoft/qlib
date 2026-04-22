import TimelineModel, { TimelineOption } from './TimelineModel.js';
import { DataFormatMixin } from '../../model/mixin/dataFormat.js';
import SeriesData from '../../data/SeriesData.js';
export interface SliderTimelineOption extends TimelineOption {
}
declare class SliderTimelineModel extends TimelineModel {
    static type: string;
    type: string;
    /**
     * @protected
     */
    static defaultOption: SliderTimelineOption;
}
interface SliderTimelineModel extends DataFormatMixin {
    getData(): SeriesData<SliderTimelineModel>;
}
export default SliderTimelineModel;
