// Location: /src/components/charts/RiskHeatmap/index.tsx
// Risk Heatmap - Thermal Vision for Your Portfolio's Soul

import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Flame, AlertOctagon } from 'lucide-react';
import * as d3 from 'd3';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';

interface RiskCell {
  position: string;
  metric: string;
  value: number;
  normalizedValue: number;
}

export const RiskHeatmap: React.FC = () => {
  const { metrics, strategies } = useDashboardStore();
  const theme = useThemeService();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    // Clear previous
    d3.select(svgRef.current).selectAll("*").remove();

    const width = 400;
    const height = 300;
    const margin = { top: 50, right: 20, bottom: 50, left: 100 };

    // Risk metrics to display
    const riskMetrics = ['Leverage', 'Volatility', 'Correlation', 'Liquidity', 'Exposure'];
    const positions = [
      ...Object.keys(metrics.positionsPnL),
      ...strategies.filter(s => s.isActive).map(s => s.codename.split(' ')[0])
    ].slice(0, 6);

    // Generate heatmap data
    const data: RiskCell[] = [];
    positions.forEach((position, i) => {
      riskMetrics.forEach((metric, j) => {
        const baseValue = 
          metric === 'Leverage' ? metrics.leverage / 10 :
          metric === 'Volatility' ? Math.random() :
          metric === 'Correlation' ? 0.3 + Math.random() * 0.5 :
          metric === 'Liquidity' ? 1 - Math.random() * 0.4 :
          Math.random();
        
        // Add position-specific variations
        const value = baseValue * (0.7 + Math.random() * 0.6);
        
        data.push({
          position,
          metric,
          value: value * 100,
          normalizedValue: Math.min(1, Math.max(0, value))
        });
      });
    });

    const svg = d3.select(svgRef.current)
      .attr("viewBox", `0 0 ${width} ${height}`);

    // Color scale - from safe to danger
    const colorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator(t => {
        if (t < 0.33) return d3.interpolateRgb(theme.colors.success, theme.colors.warning)(t * 3);
        if (t < 0.66) return d3.interpolateRgb(theme.colors.warning, theme.colors.danger)((t - 0.33) * 3);
        return theme.colors.danger;
      });

    // Scales
    const xScale = d3.scaleBand()
      .domain(positions)
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const yScale = d3.scaleBand()
      .domain(riskMetrics)
      .range([margin.top, height - margin.bottom])
      .padding(0.1);

    // Add cells
    const cells = svg.selectAll("rect")
      .data(data)
      .enter()
      .append("rect")
      .attr("x", d => xScale(d.position)!)
      .attr("y", d => yScale(d.metric)!)
      .attr("width", xScale.bandwidth())
      .attr("height", yScale.bandwidth())
      .attr("fill", d => colorScale(d.normalizedValue))
      .attr("stroke", theme.colors.primary)
      .attr("stroke-width", 0.5)
      .attr("opacity", 0.9)
      .style("cursor", "pointer");

    // Add glow effect to high-risk cells
    cells.filter(d => d.normalizedValue > 0.7)
      .style("filter", `drop-shadow(0 0 10px ${theme.colors.danger})`);

    // Add text values
    svg.selectAll("text.value")
      .data(data)
      .enter()
      .append("text")
      .attr("class", "value")
      .attr("x", d => xScale(d.position)! + xScale.bandwidth() / 2)
      .attr("y", d => yScale(d.metric)! + yScale.bandwidth() / 2)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("fill", d => d.normalizedValue > 0.5 ? "white" : theme.colors.primary)
      .attr("font-size", "10")
      .attr("font-weight", "bold")
      .attr("font-family", "monospace")
      .text(d => d.value.toFixed(0));

    // X axis
    svg.append("g")
      .attr("transform", `translate(0,${margin.top})`)
      .selectAll("text")
      .data(positions)
      .enter()
      .append("text")
      .attr("x", d => xScale(d)! + xScale.bandwidth() / 2)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .attr("fill", theme.colors.primary)
      .attr("font-size", "11")
      .attr("font-family", "monospace")
      .text(d => d.length > 8 ? d.substring(0, 8) + '...' : d);

    // Y axis
    svg.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .selectAll("text")
      .data(riskMetrics)
      .enter()
      .append("text")
      .attr("x", -10)
      .attr("y", d => yScale(d)! + yScale.bandwidth() / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .attr("fill", theme.colors.primary)
      .attr("font-size", "11")
      .attr("font-family", "monospace")
      .text(d => d);

    // Tooltip
    const tooltip = d3.select("body").append("div")
      .attr("class", "d3-tooltip")
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background-color", "rgba(0, 0, 0, 0.9)")
      .style("border", `1px solid ${theme.colors.primary}`)
      .style("border-radius", "4px")
      .style("padding", "8px")
      .style("font-size", "12px")
      .style("font-family", "monospace")
      .style("color", "white");

    cells
      .on("mouseover", function(event, d) {
        const risk = d.normalizedValue > 0.7 ? "HIGH RISK" : 
                    d.normalizedValue > 0.4 ? "MODERATE" : "LOW RISK";
        tooltip.style("visibility", "visible")
          .html(`
            <div style="color: ${theme.colors.primary}">${d.position}</div>
            <div style="color: ${theme.colors.info}">${d.metric}: ${d.value.toFixed(1)}%</div>
            <div style="color: ${colorScale(d.normalizedValue)}">${risk}</div>
          `);
      })
      .on("mousemove", function(event) {
        tooltip.style("top", (event.pageY - 10) + "px")
          .style("left", (event.pageX + 10) + "px");
      })
      .on("mouseout", function() {
        tooltip.style("visibility", "hidden");
      });

    return () => {
      d3.select("body").selectAll(".d3-tooltip").remove();
    };
  }, [metrics, strategies, theme]);

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.danger}66` }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.danger }}>
          <GlitchText theme={theme.colors}>Risk Thermal Matrix</GlitchText>
        </h3>
        <Flame className="w-5 h-5" style={{ color: theme.colors.danger }} />
      </div>

      <div className="relative">
        <svg ref={svgRef} className="w-full h-full" />
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: theme.colors.success }} />
            <span className="text-xs text-gray-400">Low Risk</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: theme.colors.warning }} />
            <span className="text-xs text-gray-400">Medium Risk</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: theme.colors.danger }} />
            <span className="text-xs text-gray-400">High Risk</span>
          </div>
        </div>
        <div className="flex items-center space-x-2 text-xs text-gray-500">
          <AlertOctagon className="w-4 h-4" />
          <span>Hover for details</span>
        </div>
      </div>

      {/* Risk Summary */}
      <div className="mt-4 p-3 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.danger}33` }}>
        <p className="text-xs text-gray-400">
          <span className="font-bold" style={{ color: theme.colors.danger }}>Thermal Analysis:</span> Hotter 
          zones indicate higher risk exposure. Concentrated heat patterns suggest correlated risks that could 
          cascade in a market downturn. Diversify to cool the matrix.
        </p>
      </div>
    </motion.div>
  );
};
