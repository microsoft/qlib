import { Dictionary, WithThisType } from './types';
export declare type EventCallbackSingleParam<EvtParam = any> = EvtParam extends any ? (params: EvtParam) => boolean | void : never;
export declare type EventCallback<EvtParams = any[]> = EvtParams extends any[] ? (...args: EvtParams) => boolean | void : never;
export declare type EventQuery = string | Object;
declare type CbThis<Ctx, Impl> = unknown extends Ctx ? Impl : Ctx;
declare type DefaultEventDefinition = Dictionary<EventCallback<any[]>>;
export interface EventProcessor<EvtDef = DefaultEventDefinition> {
    normalizeQuery?: (query: EventQuery) => EventQuery;
    filter?: (eventType: keyof EvtDef, query: EventQuery) => boolean;
    afterTrigger?: (eventType: keyof EvtDef) => void;
}
export default class Eventful<EvtDef extends DefaultEventDefinition = DefaultEventDefinition> {
    private _$handlers;
    protected _$eventProcessor: EventProcessor<EvtDef>;
    constructor(eventProcessors?: EventProcessor<EvtDef>);
    on<Ctx, EvtNm extends keyof EvtDef>(event: EvtNm, handler: WithThisType<EvtDef[EvtNm], CbThis<Ctx, this>>, context?: Ctx): this;
    on<Ctx, EvtNm extends keyof EvtDef>(event: EvtNm, query: EventQuery, handler: WithThisType<EvtDef[EvtNm], CbThis<Ctx, this>>, context?: Ctx): this;
    isSilent(eventName: keyof EvtDef): boolean;
    off(eventType?: keyof EvtDef, handler?: Function): this;
    trigger<EvtNm extends keyof EvtDef>(eventType: EvtNm, ...args: Parameters<EvtDef[EvtNm]>): this;
    triggerWithContext(type: keyof EvtDef, ...args: any[]): this;
}
export {};
