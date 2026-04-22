import { HashMap } from 'zrender/lib/core/util.js';
import { Task, TaskContext } from './task.js';
import GlobalModel from '../model/Global.js';
import ExtensionAPI from './ExtensionAPI.js';
import { StageHandlerInternal, StageHandlerOverallReset, StageHandler, Payload, StageHandlerReset, StageHandlerPlan, StageHandlerProgressExecutor, SeriesLargeOptionMixin, SeriesOption } from '../util/types.js';
import { EChartsType } from './echarts.js';
import SeriesModel from '../model/Series.js';
import ChartView from '../view/Chart.js';
import SeriesData from '../data/SeriesData.js';
export declare type GeneralTask = Task<TaskContext>;
export declare type SeriesTask = Task<SeriesTaskContext>;
export declare type OverallTask = Task<OverallTaskContext> & {
    agentStubMap?: HashMap<StubTask>;
};
export declare type StubTask = Task<StubTaskContext> & {
    agent?: OverallTask;
};
export declare type Pipeline = {
    id: string;
    head: GeneralTask;
    tail: GeneralTask;
    threshold: number;
    progressiveEnabled: boolean;
    blockIndex: number;
    step: number;
    count: number;
    currentTask?: GeneralTask;
    context?: PipelineContext;
};
export declare type PipelineContext = {
    progressiveRender: boolean;
    modDataCount: number;
    large: boolean;
};
declare type PerformStageTaskOpt = {
    block?: boolean;
    setDirty?: boolean;
    visualType?: StageHandlerInternal['visualType'];
    dirtyMap?: HashMap<any>;
};
export interface SeriesTaskContext extends TaskContext {
    model?: SeriesModel;
    data?: SeriesData;
    view?: ChartView;
    ecModel?: GlobalModel;
    api?: ExtensionAPI;
    useClearVisual?: boolean;
    plan?: StageHandlerPlan;
    reset?: StageHandlerReset;
    scheduler?: Scheduler;
    payload?: Payload;
    resetDefines?: StageHandlerProgressExecutor[];
}
interface OverallTaskContext extends TaskContext {
    ecModel: GlobalModel;
    api: ExtensionAPI;
    overallReset: StageHandlerOverallReset;
    scheduler: Scheduler;
    payload?: Payload;
}
interface StubTaskContext extends TaskContext {
    model: SeriesModel;
    overallProgress: boolean;
}
declare class Scheduler {
    readonly ecInstance: EChartsType;
    readonly api: ExtensionAPI;
    unfinished: boolean;
    private _dataProcessorHandlers;
    private _visualHandlers;
    private _allHandlers;
    private _stageTaskMap;
    private _pipelineMap;
    constructor(ecInstance: EChartsType, api: ExtensionAPI, dataProcessorHandlers: StageHandlerInternal[], visualHandlers: StageHandlerInternal[]);
    restoreData(ecModel: GlobalModel, payload: Payload): void;
    getPerformArgs(task: GeneralTask, isBlock?: boolean): {
        step: number;
        modBy: number;
        modDataCount: number;
    };
    getPipeline(pipelineId: string): Pipeline;
    /**
     * Current, progressive rendering starts from visual and layout.
     * Always detect render mode in the same stage, avoiding that incorrect
     * detection caused by data filtering.
     * Caution:
     * `updateStreamModes` use `seriesModel.getData()`.
     */
    updateStreamModes(seriesModel: SeriesModel<SeriesOption & SeriesLargeOptionMixin>, view: ChartView): void;
    restorePipelines(ecModel: GlobalModel): void;
    prepareStageTasks(): void;
    prepareView(view: ChartView, model: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    performDataProcessorTasks(ecModel: GlobalModel, payload?: Payload): void;
    performVisualTasks(ecModel: GlobalModel, payload?: Payload, opt?: PerformStageTaskOpt): void;
    private _performStageTasks;
    performSeriesTasks(ecModel: GlobalModel): void;
    plan(): void;
    updatePayload(task: Task<SeriesTaskContext | OverallTaskContext>, payload: Payload | 'remain'): void;
    private _createSeriesStageTask;
    private _createOverallStageTask;
    private _pipe;
    static wrapStageHandler(stageHandler: StageHandler | StageHandlerOverallReset, visualType: StageHandlerInternal['visualType']): StageHandlerInternal;
}
export default Scheduler;
