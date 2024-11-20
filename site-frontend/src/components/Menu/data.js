import React, { useState, useEffect } from 'react'
import Settings from '../../images/Union.png'
import { Helmet } from 'react-helmet'
import graph from '../../images/stock.png'
import './data.css'
const Data = (props) => {
  const[ModelActive, setModelActive] = useState(0);
const[DataActive, setDataActive] = useState(0);
const[ResultsActive,setResultsActive] = useState(0);
useEffect(() => {
  setModelActive(1);
  setDataActive(0);
  setResultsActive(0);
}, []);
function Model(){
    setModelActive(1);
    setDataActive(0);
    setResultsActive(0);
};
function Results(){
 setResultsActive(1);
 setDataActive(0);
 setModelActive(0);
}
function LiveTrading(){
  console.log("B");
 }
 function BigData(){
    setDataActive(1);
    setResultsActive(0);
    setModelActive(0);
 }
  return (

      <div>

        <div className="data-menu2">
          <div className="data-menulist">
                  <span className="data-text12">
                    <span onClick = {BigData}>Data</span>
                  </span>
                  <span className="data-text12 ">
                    <span onClick = {Model}>Model</span>
                  </span>
                  <span className="data-text12">
                    <span onClick = {Results}>Results</span>
                  </span>
                  <span className="data-text12">
                    <span >Live Trading</span>
                  </span>
          </div>
        </div>
        <div className="data-union1">
          <div className = "data-union2">
            <img src = {Settings} className = "data-settings"></img>
          </div>
        </div>
        <div clasName = 'LogoContent'>
        <span className="data-text18 M3bodylarge">
          <span>AI Finance Logo Name</span>
        </span>
        </div>
        <div className = "Column_2" style = {{zIndex:DataActive}}>
        <div className="data-input-field10">
          <span className="data-text20 BodyBase">
            <span>Ticker</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field11">
          <span className="data-text22 BodyBase">
            <span className = "label-size">Start Date</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field12">
          <span className="data-text24 BodyBase">
            <span>End Date</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field13">
          <span className="data-text26 BodyBase">
            <span>Interval</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field14">
          <span className="data-text28 BodyBase">
            <span>Price point</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field15">
          <span className="data-text30 BodyBase">
            <span>Dataset</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        </div>
        <span className="data-text32 M3bodylarge" style = {{zIndex:DataActive}}>
          <span>Data choices</span>
        </span>
        <div className = "Column_3" style = {{zIndex:(ModelActive*200)}}>
        <div className="data-input-field16">
          <span className="data-text34 BodyBase">
            <span>Imputation method</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field17">
          <span className="data-text36 BodyBase">
            <span>Label</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field18">
          <span className="data-text38 BodyBase">
            <span>Label</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field19">
          <span className="data-text40 BodyBase">
            <span>Label</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field20">
          <span className="data-text42 BodyBase">
            <span>Label</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field21">
          <span className="data-text44">
            <span>
              Scale
              <span
                
              />
            </span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <span className="data-text46 M3bodylarge">
          <span>Data Scaling and Imputation</span>
        </span>
        </div>
        <div className = "Column_4">

        </div>
        <div className = "data-rectangle8">

        </div>
        <span className="data-text48 M3bodylarge">
          <span>Explain and Tips on Hover?</span>
        </span>
        <div className = "Model_Choice"style = {{zIndex:ModelActive}}>
        <span className="Model-Header">
          <span>Model Choices</span>
        </span>
        <div className="data-input-field10">
          <span className="data-text20 BodyBase">
            <span>HyperParam 1</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field11">
          <span className="data-text22 BodyBase">
            <span className = "label-size">HyperParam 2</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field12">
          <span className="data-text24 BodyBase">
            <span>HyperParam 3</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field13">
          <span className="data-text26 BodyBase">
            <span>HyperParam 4</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field14">
          <span className="data-text28 BodyBase">
            <span>HyperParam 5</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field15">
          <span className="data-text30 BodyBase">
            <span>Model</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        </div>
        <img src = {graph} className = "graph-settings" style = {{zIndex:ResultsActive*1000}}></img>
        <div className = "Column_2" style = {{zIndex:ResultsActive}}>
        <div className = "results-page">
            <span className = "ticker-1" >
              Ticker 1 
            </span>
            <ul>
              <li>Dates</li>
              <li>Models</li>
              <li>List item</li>
              <li>List item</li>
              <li>List item</li>
        </ul>
        <div className = "generate-button">
          <span className = "generate-text">
            Generate Graph
          </span>
        </div>
        <div className = "generate-table">
          <span className = "generate-text">
            Generate Table
          </span>
        </div>
        </div>
        <div className = "results-page2">
        <span className = "ticker-1" >
              Ticker 1 
            </span>
            <ul>
              <li>Dates</li>
              <li>Models</li>
              <li>List item</li>
              <li>List item</li>
              <li>List item</li>
        </ul>
        <div className = "generate-button">
          <span className = "generate-text">
            Generate Graph
          </span>
        </div>
        <div className = "generate-table">
          <span className = "generate-text">
            Generate Table
          </span>
        </div>
        </div>
        <span className="Results-Header" style = {{zIndex:ResultsActive}}>
          <span>Summary</span>
        </span>
        </div>
      </div>
  )
}

export default Data
