#!/usr/bin/env python3

#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
	page_title="SAMARTH Project Analysis Dashboard",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
	.main {background-color: #f5f7fa;}
	.stTabs [data-baseweb="tab-list"] {gap: 8px;}
	.stTabs [data-baseweb="tab"] {
		background-color: #e8eef5;
		border-radius: 4px 4px 0 0;
		padding: 10px 20px;
		font-weight: 600;
	}
	.stTabs [aria-selected="true"] {
		background-color: #1f77b4;
		color: white;
	}
	div[data-testid="metric-container"] {
		background-color: #ffffff;
		border: 1px solid #e0e0e0;
		padding: 15px;
		border-radius: 8px;
		box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
	}
	h1 {color: #1f77b4; font-weight: 700;}
	h3 {color: #2c3e50;}
</style>
""", unsafe_allow_html=True)


def create_abbreviations(names, max_length=35):
	"""
	Create abbreviations for long names and return mapping
	
	Args:
		names: List or Series of names
		max_length: Maximum length before abbreviation (default 35)
	
	Returns:
		tuple: (abbreviated_names, legend_df)
	"""
	abbrev_map = {}
	abbreviated = []
	
	for name in names:
		name_str = str(name)
		if len(name_str) > max_length:
			# Create abbreviation
			words = name_str.split()
			if len(words) > 4:
				# Use acronym for multi-word names
				abbrev = ''.join([w[0].upper() for w in words[:6]])
				# Add first word for context
				abbrev = f"{words[0][:10]}... ({abbrev})"
			elif len(words) > 1:
				# Use first and last word
				abbrev = f"{words[0][:15]}...{words[-1][:10]}"
			else:
				# Just truncate
				abbrev = name_str[:max_length-3] + "..."
				
			abbrev_map[abbrev] = name_str
			abbreviated.append(abbrev)
		else:
			abbreviated.append(name_str)
			
	# Create legend DataFrame
	if abbrev_map:
		legend_df = pd.DataFrame([
			{'Abbreviation': k, 'Full Name': v} 
			for k, v in abbrev_map.items()
		])
	else:
		legend_df = None
		
	return abbreviated, legend_df

def display_chart_with_legend(fig, legend_df, title="📖 Full Names Legend"):
	"""Display chart and legend if abbreviations were used"""
	st.plotly_chart(fig, use_container_width=True)
	
	if legend_df is not None and len(legend_df) > 0:
		with st.expander(f"{title} ({len(legend_df)} abbreviations)"):
			st.dataframe(legend_df, use_container_width=True, hide_index=True)
			
@st.cache_data
def load_all_sheets_from_uploaded_file(uploaded_file):
	"""Load all sheets from uploaded Excel file"""
	try:
		excel_file = pd.ExcelFile(uploaded_file)
		sheets = {}
		
		for sheet_name in excel_file.sheet_names:
			df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
			df.columns = df.columns.str.strip()
			
			# Convert numeric columns
			numeric_cols = ['Project Cost', 'Total Fund Released', 'Total Withdrawal Issued',
							'Actual Trained', 'Pass', 'Placed', 'In- training', 'Target']
			
			for col in numeric_cols:
				if col in df.columns:
					df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
					
			# Calculate percentages
			if 'Total Fund Released' in df.columns and 'Project Cost' in df.columns:
				df['Fund Released %'] = np.where(df['Project Cost'] > 0,
												(df['Total Fund Released'] / df['Project Cost'] * 100), 0)
				
			if 'Total Withdrawal Issued' in df.columns and 'Project Cost' in df.columns:
				df['Withdrawal %'] = np.where(df['Project Cost'] > 0,
											(df['Total Withdrawal Issued'] / df['Project Cost'] * 100), 0)
				
			if 'Pass' in df.columns and 'Actual Trained' in df.columns:
				df['Pass %'] = np.where(df['Actual Trained'] > 0,
										(df['Pass'] / df['Actual Trained'] * 100), 0)
				
			if 'Placed' in df.columns and 'Actual Trained' in df.columns:
				df['Placed %'] = np.where(df['Actual Trained'] > 0,
										(df['Placed'] / df['Actual Trained'] * 100), 0)
				
			# Date columns
			if 'EC Date' in df.columns:
				df['EC Date'] = pd.to_datetime(df['EC Date'], errors='coerce')
				
			# Fill NaN in text columns
			text_cols = ['Type', 'Status', 'IP Name']
			for col in text_cols:
				if col in df.columns:
					df[col] = df[col].fillna('Unknown')
					
			sheets[sheet_name] = df
			
		return sheets
	
	except Exception as e:
		st.error(f"Error loading file: {str(e)}")
		return None
	try:
		excel_file = pd.ExcelFile(file_path)
		sheets = {}
		
		for sheet_name in excel_file.sheet_names:
			df = pd.read_excel(file_path, sheet_name=sheet_name)
			df.columns = df.columns.str.strip()
			
			# Convert numeric columns
			numeric_cols = ['Project Cost', 'Total Fund Released', 'Total Withdrawal Issued',
							'Actual Trained', 'Pass', 'Placed', 'In- training', 'Target']
			
			for col in numeric_cols:
				if col in df.columns:
					df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
					
			# Calculate percentages
			if 'Total Fund Released' in df.columns and 'Project Cost' in df.columns:
				df['Fund Released %'] = np.where(df['Project Cost'] > 0,
												(df['Total Fund Released'] / df['Project Cost'] * 100), 0)
				
			if 'Total Withdrawal Issued' in df.columns and 'Project Cost' in df.columns:
				df['Withdrawal %'] = np.where(df['Project Cost'] > 0,
											(df['Total Withdrawal Issued'] / df['Project Cost'] * 100), 0)
				
			if 'Pass' in df.columns and 'Actual Trained' in df.columns:
				df['Pass %'] = np.where(df['Actual Trained'] > 0,
										(df['Pass'] / df['Actual Trained'] * 100), 0)
				
			if 'Placed' in df.columns and 'Actual Trained' in df.columns:
				df['Placed %'] = np.where(df['Actual Trained'] > 0,
										(df['Placed'] / df['Actual Trained'] * 100), 0)
				
			# Date columns
			if 'EC Date' in df.columns:
				df['EC Date'] = pd.to_datetime(df['EC Date'], errors='coerce')
				
			# Fill NaN in text columns
			text_cols = ['Type', 'Status', 'IP Name']
			for col in text_cols:
				if col in df.columns:
					df[col] = df[col].fillna('Unknown')
					
			sheets[sheet_name] = df
			
		print(f"Loaded {len(sheets)} sheets successfully")
		return sheets
	
	except Exception as e:
		st.error(f"Error loading file: {str(e)}")
		return None
	
def create_executive_kpis(main_df):
	"""Executive KPI Dashboard"""
	col1, col2, col3, col4, col5 = st.columns(5)  # Changed from 6 to 5
	
	total_projects = len(main_df)
	total_cost = main_df['Project Cost'].sum() / 1e7
	total_released = main_df['Total Fund Released'].sum() / 1e7  # Keep for calculation but don't display
	total_trained = int(main_df['Actual Trained'].sum())
	total_pass = int(main_df['Pass'].sum())
	total_placed = int(main_df['Placed'].sum())
	
	with col1:
		st.metric("Total Projects", f"{total_projects:,}")
	with col2:
		st.metric("Project Cost (₹ Cr)", f"{total_cost:.2f}")
	with col3:
		st.metric("Total Trained", f"{total_trained:,}")
	with col4:
		pass_rate = (total_pass / total_trained * 100) if total_trained > 0 else 0
		st.metric("Pass Rate", f"{pass_rate:.1f}%")
	with col5:
		place_rate = (total_placed / total_trained * 100) if total_trained > 0 else 0
		st.metric("Placement Rate", f"{place_rate:.1f}%")
		
def analyze_sop_category_1(main_df, top_n=20):
	"""Category 1: Total Closed Projects (100% Fund Released & Withdrawal)"""
	st.subheader("Category 1: Total Closed Projects (100% Utilization)")
	
	# Filter: Fund Released % = 100 AND Withdrawal % = 100
	closed_df = main_df[
		(main_df['Fund Released %'] == 100) & 
		(main_df['Withdrawal %'] == 100)
	]
	
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Closed Projects (100% Util)", f"{len(closed_df):,}")
	with col2:
		st.metric("Total Cost (₹ Cr)", f"{closed_df['Project Cost'].sum()/1e7:.2f}")
	with col3:
		st.metric("Total Trained", f"{int(closed_df['Actual Trained'].sum()):,}")
		
	if len(closed_df) > 0:
		st.markdown("---")
		
		# Sort metric selector
		sort_metric = st.selectbox(
			"Sort by:",
			options=['Actual Trained', 'Project Cost', 'Pass', 'Placed', 'Pass %'],
			index=0,
			key="closed_sort"
		)
		
		# Get top projects and create abbreviations
		top_closed = closed_df.nlargest(top_n, sort_metric).copy()
		
		if 'IP Name' in top_closed.columns:
			abbreviated_names, legend_df = create_abbreviations(top_closed['IP Name'].tolist())
			top_closed['IP Name Abbrev'] = abbreviated_names
			
			# Visualization
			fig = px.bar(
				top_closed,
				x='IP Name Abbrev', 
				y=sort_metric,
				title=f"Top {top_n} Closed Projects by {sort_metric}",
				color='Pass %', 
				color_continuous_scale='Greens',
				text=sort_metric,
				hover_data={'IP Name': True, 'IP Name Abbrev': False}
			)
			fig.update_layout(
				xaxis={'tickangle': -45, 'title': 'Implementing Partner'},
				height=500
			)
			fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
			
			# Display with legend
			display_chart_with_legend(fig, legend_df, "📖 Full IP Names")
			
		# Additional charts
		st.markdown("---")
		col1, col2 = st.columns(2)
		
		with col1:
			fig2 = px.histogram(closed_df, x='Pass %', nbins=30,
								title="Pass Rate Distribution",
								color_discrete_sequence=['#2ecc71'])
			fig2.add_vline(x=closed_df['Pass %'].mean(), line_dash="dash", 
							line_color="red", annotation_text=f"Avg: {closed_df['Pass %'].mean():.1f}%")
			st.plotly_chart(fig2, use_container_width=True)
			
		with col2:
			type_dist = closed_df.groupby('Type').size().reset_index(name='Count')
			fig3 = px.pie(type_dist, values='Count', names='Type',
						title="Closed Projects by Type", hole=0.4)
			st.plotly_chart(fig3, use_container_width=True)
	else:
		st.info("No projects with 100% fund released and withdrawal found.")
		
def analyze_sop_category_4(main_df, top_n=20):
	"""Category 4: Projects eligible for closure (>86% AND <100%)"""
	st.subheader("Category 4: Eligible for Closure (>86% and <100%)")
	
	# Filter: Fund Released > 86 AND < 100, Withdrawal > 86 AND < 100
	eligible_df = main_df[
		(main_df['Fund Released %'] > 86) & (main_df['Fund Released %'] < 100) &
		(main_df['Withdrawal %'] > 86) & (main_df['Withdrawal %'] < 100)
	]
	
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Eligible for Closure", f"{len(eligible_df):,}")
	with col2:
		st.metric("Total Cost (₹ Cr)", f"{eligible_df['Project Cost'].sum()/1e7:.2f}")
	with col3:
		st.metric("Avg Fund Released", f"{eligible_df['Fund Released %'].mean():.1f}%")
	with col4:
		st.metric("Total Trained", f"{int(eligible_df['Actual Trained'].sum()):,}")
		
	if len(eligible_df) > 0:
		# Scatter plot
		display_df = eligible_df.nlargest(top_n, 'Actual Trained') if len(eligible_df) > top_n else eligible_df
		
		fig = px.scatter(display_df, x='Fund Released %', y='Withdrawal %',
						size='Actual Trained', color='Pass %',
						hover_name='IP Name',
						title=f"Top {min(top_n, len(eligible_df))} Eligible Projects",
						color_continuous_scale='Greens')
		fig.add_hline(y=86, line_dash="dash", line_color="red", annotation_text="86%")
		fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="100%")
		fig.add_vline(x=86, line_dash="dash", line_color="red")
		fig.add_vline(x=100, line_dash="dash", line_color="green")
		st.plotly_chart(fig, use_container_width=True)
		
		# Status breakdown
		col_a, col_b = st.columns(2)
		
		with col_a:
			status_breakdown = eligible_df['Status'].value_counts().head(10).reset_index()
			status_breakdown.columns = ['Status', 'Count']
			
			abbrev_status, status_legend = create_abbreviations(status_breakdown['Status'].tolist(), max_length=40)
			status_breakdown['Status Abbrev'] = abbrev_status
			
			fig2 = px.bar(status_breakdown, x='Status Abbrev', y='Count',
						title="Top 10 Status Categories")
			fig2.update_layout(xaxis={'tickangle': -45})
			display_chart_with_legend(fig2, status_legend, "📖 Full Status Names")
			
		with col_b:
			if 'Documents Status' in eligible_df.columns:
				doc_status = eligible_df['Documents Status'].value_counts().head(10).reset_index()
				doc_status.columns = ['Document Status', 'Count']
				
				fig3 = px.pie(doc_status, values='Count', names='Document Status',
							title="Top 10 Document Status")
				st.plotly_chart(fig3, use_container_width=True)
	else:
		st.info("No projects found eligible for closure (>86% and <100%).")
		
def analyze_sop_category_5(main_df, top_n=20):
	"""Category 5: Recovery Projects (>100% Utilization)"""
	st.subheader("Category 5: Recovery Projects (>100% Utilization)")
	
	# Filter: Fund Released > 100 OR Withdrawal > 100
	recovery_df = main_df[
		(main_df['Fund Released %'] > 100) | 
		(main_df['Withdrawal %'] > 100)
	]
	
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Recovery Projects", f"{len(recovery_df):,}")
	with col2:
		excess_amount = recovery_df['Total Fund Released'].sum() - recovery_df['Project Cost'].sum()
		st.metric("Excess Released (₹ Cr)", f"{excess_amount/1e7:.2f}")
	with col3:
		avg_release = recovery_df['Fund Released %'].mean()
		st.metric("Avg Fund Released", f"{avg_release:.1f}%")
	with col4:
		st.metric("Total Cost (₹ Cr)", f"{recovery_df['Project Cost'].sum()/1e7:.2f}")
		
	if len(recovery_df) > 0:
		col_a, col_b = st.columns(2)
		
		with col_a:
			# Top N recovery projects
			top_recovery = recovery_df.nlargest(min(top_n, len(recovery_df)), 'Fund Released %').copy()
			
			if 'IP Name' in top_recovery.columns:
				abbrev_names, legend_df = create_abbreviations(top_recovery['IP Name'].tolist())
				top_recovery['IP Name Abbrev'] = abbrev_names
				
				fig = px.bar(top_recovery, x='IP Name Abbrev', y='Fund Released %',
							title=f"Top {len(top_recovery)} Projects by Over-Utilization",
							color='Withdrawal %', color_continuous_scale='Reds',
							text='Fund Released %')
				fig.update_layout(xaxis={'tickangle': -45}, height=400)
				fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
				display_chart_with_legend(fig, legend_df, "Full IP Names")
				
		with col_b:
			fig2 = px.scatter(recovery_df, x='Fund Released %', y='Withdrawal %',
							size='Project Cost', color='Actual Trained',
							hover_name='IP Name',
							title="Fund Released vs Withdrawal (All Recovery)",
							color_continuous_scale='Viridis')
			fig2.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100%")
			fig2.add_vline(x=100, line_dash="dash", line_color="red")
			st.plotly_chart(fig2, use_container_width=True)
			
		# SPOC-wise analysis
		if 'SPOC' in recovery_df.columns:
			st.markdown("---")
			st.subheader("SPOC-wise Recovery Analysis")
			
			spoc_recovery = recovery_df.groupby('SPOC').agg({
				'Project ID': 'count',
				'Total Fund Released': 'sum',
				'Project Cost': 'sum'
			}).reset_index()
			spoc_recovery.columns = ['SPOC', 'Projects', 'Total Fund Released', 'Project Cost']
			spoc_recovery['Excess (₹ Cr)'] = (spoc_recovery['Total Fund Released'] - spoc_recovery['Project Cost']) / 1e7
			spoc_recovery = spoc_recovery.nlargest(10, 'Excess (₹ Cr)')
			
			st.dataframe(spoc_recovery[['SPOC', 'Projects', 'Excess (₹ Cr)']], 
						use_container_width=True, hide_index=True)
			
		# Data table
		with st.expander("View Recovery Projects Data"):
			display_cols = ['IP Name', 'Project Cost', 'Total Fund Released', 
							'Fund Released %', 'Withdrawal %', 'Actual Trained', 'Status', 'SPOC']
			available_cols = [col for col in display_cols if col in recovery_df.columns]
			st.dataframe(recovery_df[available_cols], use_container_width=True)
	else:
		st.success("No over-utilized projects found!")
		
def analyze_sop_category_6(main_df, top_n=20):
	"""Category 6: Performance Analysis - Target vs Trained vs Pass vs Placed"""
	st.subheader("Category 6: Training Performance Analysis")
	
	# Filter projects with actual training
	performance_df = main_df[main_df['Actual Trained'] > 0].copy()
	
	# Calculate performance metrics
	performance_df['Target Achievement %'] = np.where(
		performance_df['Target'] > 0,
		(performance_df['Actual Trained'] / performance_df['Target'] * 100), 0
	)
	
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		target_total = performance_df['Target'].sum()
		st.metric("Total Target", f"{int(target_total):,}")
	with col2:
		trained_total = performance_df['Actual Trained'].sum()
		achievement = (trained_total / target_total * 100) if target_total > 0 else 0
		st.metric("Actual Trained", f"{int(trained_total):,}", delta=f"{achievement:.1f}% of target")
	with col3:
		pass_total = performance_df['Pass'].sum()
		pass_rate = (pass_total / trained_total * 100) if trained_total > 0 else 0
		st.metric("Total Passed", f"{int(pass_total):,}", delta=f"{pass_rate:.1f}% pass rate")
	with col4:
		placed_total = performance_df['Placed'].sum()
		placement_rate = (placed_total / trained_total * 100) if trained_total > 0 else 0
		st.metric("Total Placed", f"{int(placed_total):,}", delta=f"{placement_rate:.1f}% placement")
		
	if len(performance_df) > 0:
		# Top Performers
		st.markdown("---")
		st.subheader("Top Performing Projects")
		
		# Calculate performance score
		performance_df['Performance Score'] = (
			performance_df['Target Achievement %'] * 0.3 +
			performance_df['Pass %'] * 0.4 +
			performance_df['Placed %'] * 0.3
		)
		
		top_performers = performance_df.nlargest(top_n, 'Performance Score').copy()
		
		# Create abbreviations
		if 'IP Name' in top_performers.columns:
			abbreviated_names, legend_df = create_abbreviations(top_performers['IP Name'].tolist())
			top_performers['IP Name Abbrev'] = abbreviated_names
			
			fig = go.Figure()
			fig.add_trace(go.Bar(name='Target', x=top_performers['IP Name Abbrev'], 
								y=top_performers['Target'], marker_color='lightblue'))
			fig.add_trace(go.Bar(name='Trained', x=top_performers['IP Name Abbrev'], 
								y=top_performers['Actual Trained'], marker_color='blue'))
			fig.add_trace(go.Bar(name='Pass', x=top_performers['IP Name Abbrev'], 
								y=top_performers['Pass'], marker_color='green'))
			fig.add_trace(go.Bar(name='Placed', x=top_performers['IP Name Abbrev'], 
								y=top_performers['Placed'], marker_color='purple'))
			
			fig.update_layout(
				title=f"Top {top_n} Performers: Training Funnel",
				barmode='group', 
				xaxis={'tickangle': -45, 'title': 'Implementing Partner'}, 
				height=500
			)
			
			display_chart_with_legend(fig, legend_df, "Full IP Names")
			
		# Performance quadrant
		st.markdown("---")
		col_a, col_b = st.columns(2)
		
		with col_a:
			fig2 = px.scatter(performance_df, x='Pass %', y='Placed %',
							size='Actual Trained', color='Type',
							hover_name='IP Name',
							title="Pass Rate vs Placement Rate")
			
			median_pass = performance_df['Pass %'].median()
			median_placed = performance_df['Placed %'].median()
			fig2.add_hline(y=median_placed, line_dash="dash", line_color="gray", opacity=0.5)
			fig2.add_vline(x=median_pass, line_dash="dash", line_color="gray", opacity=0.5)
			
			st.plotly_chart(fig2, use_container_width=True)
			
		with col_b:
			fig3 = px.scatter(performance_df, x='Target Achievement %', y='Pass %',
							size='Project Cost', color='Placed %',
							hover_name='IP Name',
							title="Target Achievement vs Pass Rate",
							color_continuous_scale='RdYlGn')
			fig3.add_vline(x=100, line_dash="dash", line_color="red", 
							annotation_text="100% Target")
			st.plotly_chart(fig3, use_container_width=True)
			
		# Under-performers
		st.markdown("---")
		st.subheader("Under-performing Projects")
		
		underperformers = performance_df[
			(performance_df['Target Achievement %'] < 50) |
			(performance_df['Pass %'] < 70) |
			(performance_df['Placed %'] < 30)
		]
		
		col1, col2, col3 = st.columns(3)
		with col1:
			low_target = len(performance_df[performance_df['Target Achievement %'] < 50])
			st.metric("Low Target Achievement (<50%)", f"{low_target:,}")
		with col2:
			low_pass = len(performance_df[performance_df['Pass %'] < 70])
			st.metric("Low Pass Rate (<70%)", f"{low_pass:,}")
		with col3:
			low_placement = len(performance_df[performance_df['Placed %'] < 30])
			st.metric("Low Placement (<30%)", f"{low_placement:,}")

def analyze_sop_category_2(main_df):
	"""Category 2: Projects empaneled before March 2025 with zero training"""
	st.subheader("Category 2: Pre-March 2025 Projects with Zero Training")
	
	# Filter: EC Date < March 2025
	pre_march_df = main_df[main_df['EC Date'] < '2025-03-01']
	
	# Sub-filter: In-training and Actual Trained = 0
	zero_training_df = pre_march_df[
		(pre_march_df['In- training'] == 0) & 
		(pre_march_df['Actual Trained'] == 0)
	]
	
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Total Pre-March 2025", f"{len(pre_march_df):,}")
	with col2:
		st.metric("Zero Training Projects", f"{len(zero_training_df):,}", 
				delta=f"{len(zero_training_df)/len(pre_march_df)*100:.1f}% of total" if len(pre_march_df) > 0 else "0%")
	with col3:
		st.metric("Unutilized Cost (₹ Cr)", f"{zero_training_df['Project Cost'].sum()/1e7:.2f}")
	with col4:
		st.metric("Fund Released (₹ Cr)", f"{zero_training_df['Total Fund Released'].sum()/1e7:.2f}")
		
	if len(zero_training_df) > 0:
		col_a, col_b = st.columns(2)
		
		with col_a:
			type_summary = zero_training_df.groupby('Type').agg({
				'Project Cost': 'sum',
				'Project ID': 'count'
			}).reset_index()
			type_summary.columns = ['Type', 'Total Cost', 'Count']
			
			fig1 = px.bar(type_summary, x='Type', y='Total Cost',
						title="Zero Training Projects by Type (₹)",
						color='Count', text='Count',
						color_continuous_scale='Reds')
			fig1.update_traces(texttemplate='%{text}', textposition='outside')
			st.plotly_chart(fig1, use_container_width=True)
			
		with col_b:
			status_summary = zero_training_df.groupby('Status').agg({
				'Project Cost': 'sum',
				'Project ID': 'count'
			}).reset_index()
			status_summary.columns = ['Status', 'Total Cost', 'Count']
			status_summary = status_summary.nlargest(10, 'Total Cost')
			
			# Abbreviate long status
			abbrev_status, status_legend = create_abbreviations(status_summary['Status'].tolist(), max_length=40)
			status_summary['Status Abbrev'] = abbrev_status
			
			fig2 = px.bar(status_summary, x='Status Abbrev', y='Total Cost',
						title="Top 10 Status Categories (₹)",
						color='Count', text='Count',
						color_continuous_scale='Oranges',
						hover_data={'Status': True, 'Status Abbrev': False})
			fig2.update_layout(xaxis={'tickangle': -45})
			fig2.update_traces(texttemplate='%{text}', textposition='outside')
			display_chart_with_legend(fig2, status_legend, "Full Status Names")
			
		# Data table
		with st.expander("View Zero Training Projects Data"):
			display_cols = ['IP Name', 'EC Date', 'Project Cost', 'Total Fund Released', 
							'Target', 'Status', 'SPOC']
			available_cols = [col for col in display_cols if col in zero_training_df.columns]
			st.dataframe(zero_training_df[available_cols], use_container_width=True)
	else:
		st.success("No projects found with zero training before March 2025!")
		
def analyze_sop_category_3(main_df):
	"""Category 3: July-Aug 2025 EC projects with zero training"""
	st.subheader("Category 3: July-Aug 2025 EC Projects with Zero Training")
	
	# Filter: EC Date in July-Aug 2025
	july_aug_df = main_df[
		(main_df['EC Date'] >= '2025-07-01') & 
		(main_df['EC Date'] <= '2025-08-31')
	]
	
	# Sub-filter: Zero training
	zero_training_df = july_aug_df[
		(july_aug_df['In- training'] == 0) & 
		(july_aug_df['Actual Trained'] == 0)
	]
	
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("July-Aug 2025 Projects", f"{len(july_aug_df):,}")
	with col2:
		st.metric("With Zero Training", f"{len(zero_training_df):,}")
	with col3:
		pct = (len(zero_training_df)/len(july_aug_df)*100) if len(july_aug_df) > 0 else 0
		st.metric("% Not Started", f"{pct:.1f}%")
		
	if len(zero_training_df) > 0:
		col_a, col_b = st.columns(2)
		
		with col_a:
			type_dist = zero_training_df.groupby('Type').size().reset_index(name='Count')
			fig1 = px.bar(type_dist, x='Type', y='Count',
						title="Projects by Type",
						color='Count', color_continuous_scale='Oranges',
						text='Count')
			fig1.update_traces(texttemplate='%{text}', textposition='outside')
			st.plotly_chart(fig1, use_container_width=True)
			
		with col_b:
			if 'Documents Status' in zero_training_df.columns:
				doc_status = zero_training_df['Documents Status'].fillna('Not Available').value_counts().head(10).reset_index()
				doc_status.columns = ['Document Status', 'Count']
				
				fig2 = px.pie(doc_status, values='Count', names='Document Status',
							title="Document Status Distribution")
				st.plotly_chart(fig2, use_container_width=True)
				
		# Data table
		with st.expander("View July-Aug 2025 Zero Training Projects"):
			display_cols = ['IP Name', 'EC Date', 'Project Cost', 'Target', 
							'Documents Status', 'Status', 'SPOC']
			available_cols = [col for col in display_cols if col in zero_training_df.columns]
			st.dataframe(zero_training_df[available_cols], use_container_width=True)
	else:
		st.success("All July-Aug 2025 projects have started training!")
		

def main():
	# Header
	st.title("SAMARTH Project Analysis Dashboard")
	st.markdown("**Ministry of Textiles, Govt of India**")
	st.markdown("*Comprehensive Analysis Following SOP Requirements*")
	st.markdown(f"Data as on: **14 January 2026** | Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p IST')}")
	st.markdown("---")
	
	# FILE UPLOADER - Add this at the very beginning
	uploaded_file = st.sidebar.file_uploader(
		"Upload SAMARTH Excel File",
		type=['xlsx'],
		help="Upload the Project-wise Status Excel file to begin analysis"
	)
	
	if uploaded_file is None:
		st.warning("⚠️ Please upload the SAMARTH project data Excel file from the sidebar to begin analysis")
		st.info("👈 Use the **'Browse files'** button in the sidebar to upload your Excel file")
		st.markdown("""
		### Expected File Format:
		- Excel file (.xlsx)
		- Multiple sheets with project data
		- Main Sheet should contain columns: IP Name, Project Cost, Actual Trained, Pass, Placed, Status, etc.
		""")
		st.stop()
		
	# Load data from uploaded file
	with st.spinner("🔄 Loading SAMARTH project data..."):
		all_sheets = load_all_sheets_from_uploaded_file(uploaded_file)
		
	if all_sheets is None:
		st.error("❌ Failed to load data. Please check the file format.")
		return
	
	# Sidebar controls
	st.sidebar.success(f"File loaded successfully!")
	st.sidebar.markdown("---")
	st.sidebar.header("Dashboard Controls")
	
	# Global Top N filter
	top_n = st.sidebar.slider(
		"Select Top N for Charts:",
		min_value=5,
		max_value=50,
		value=20,
		step=5,
		help="Adjust number of items shown in all charts across the dashboard"
	)
	
	st.sidebar.markdown("---")
	st.sidebar.header("Sheet Selection")
	st.sidebar.info(f"Total Sheets: {len(all_sheets)}")
	
	selected_sheet = st.sidebar.selectbox(
		"Select Main Sheet:",
		options=list(all_sheets.keys()),
		index=0
	)
	
	main_df = all_sheets[selected_sheet]
	
	st.sidebar.markdown("---")
	st.sidebar.markdown(f"**Current Sheet:** {selected_sheet}")
	st.sidebar.markdown(f"**Records:** {len(main_df):,}")
	st.sidebar.markdown(f"**Columns:** {len(main_df.columns)}")
	
	# Executive Summary
	st.header("Executive Summary")
	create_executive_kpis(main_df)
	
	st.markdown("---")
	
	# Create tabs
	tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
		"1️⃣ Closed Projects",
		"2️⃣ Pre-March Zero Training",
		"3️⃣ July-Aug Zero Training",
		"4️⃣ Eligible for Closure",
		"5️⃣ Recovery Projects",
		"6️⃣ Performance Analysis",
		"All Sheets Data"
	])
	
	with tab1:
		analyze_sop_category_1(main_df, top_n)
		
	with tab2:
		analyze_sop_category_2(main_df)
		
	with tab3:
		analyze_sop_category_3(main_df)
		
	with tab4:
		analyze_sop_category_4(main_df, top_n)
		
	with tab5:
		analyze_sop_category_5(main_df, top_n)
		
	with tab6:
		analyze_sop_category_6(main_df, top_n)
		
	with tab7:
		st.header("All Sheets Data Explorer")
		
		# Sheet selector
		sheet_to_view = st.selectbox(
			"Select Sheet to View:",
			options=list(all_sheets.keys())
		)
		
		df_to_display = all_sheets[sheet_to_view]
		
		# Display sheet metrics
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.metric("Total Records", f"{len(df_to_display):,}")
		with col2:
			st.metric("Total Columns", f"{len(df_to_display.columns)}")
		with col3:
			if 'Project Cost' in df_to_display.columns:
				st.metric("Total Cost (₹ Cr)", f"{df_to_display['Project Cost'].sum()/1e7:.2f}")
		with col4:
			if 'Actual Trained' in df_to_display.columns:
				st.metric("Total Trained", f"{int(df_to_display['Actual Trained'].sum()):,}")
				
		st.markdown("---")
		
		# Column selector
		st.subheader("Select Columns to Display")
		all_columns = df_to_display.columns.tolist()
		
		default_cols = ['IP Name', 'Project Cost', 'Total Fund Released', 
						'Actual Trained', 'Pass', 'Placed', 'Status']
		default_cols = [col for col in default_cols if col in all_columns]
		
		selected_columns = st.multiselect(
			"Choose columns:",
			options=all_columns,
			default=default_cols if default_cols else all_columns[:10]
		)
		
		if selected_columns:
			# Search functionality
			search_term = st.text_input("🔍 Search in data:", "")
			
			df_display = df_to_display[selected_columns].copy()
			
			if search_term:
				mask = df_to_display.astype(str).apply(
					lambda x: x.str.contains(search_term, case=False, na=False)
				).any(axis=1)
				df_display = df_to_display[mask][selected_columns]
				st.info(f"Found {len(df_display)} matching records")
				
			# Display options
			col1, col2 = st.columns(2)
			with col1:
				rows_to_show = st.selectbox("Rows to display:", 
											options=[10, 25, 50, 100, "All"], 
											index=2)
			with col2:
				if selected_columns:
					sort_column = st.selectbox("Sort by:", options=selected_columns)
					ascending = st.checkbox("Ascending order", value=False)
					if sort_column:
						df_display = df_display.sort_values(by=sort_column, ascending=ascending)
						
			# Display dataframe
			st.markdown("---")
			st.subheader("Data Table")
			
			if rows_to_show == "All":
				st.dataframe(df_display, use_container_width=True, height=600)
			else:
				st.dataframe(df_display.head(rows_to_show), use_container_width=True, height=600)
				st.caption(f"Showing {min(rows_to_show, len(df_display))} of {len(df_display):,} records")
				
			# Export options
			st.markdown("---")
			st.subheader("Export Data")
			
			col1, col2 = st.columns(2)
			
			with col1:
				csv = df_display.to_csv(index=False).encode('utf-8')
				st.download_button(
					label="Download as CSV",
					data=csv,
					file_name=f"SAMARTH_{sheet_to_view}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
					mime="text/csv"
				)
				
			with col2:
				from io import BytesIO
				buffer = BytesIO()
				with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
					df_display.to_excel(writer, sheet_name='Data', index=False)
				excel_data = buffer.getvalue()
				
				st.download_button(
					label="Download as Excel",
					data=excel_data,
					file_name=f"SAMARTH_{sheet_to_view}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
					mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
				)
		else:
			st.warning("Please select at least one column to display")
			
	# Additional Analysis
	st.markdown("---")
	st.header("Additional Insights")
	
	col1, col2 = st.columns(2)
	
	with col1:
		st.subheader("Type-wise Distribution")
		if 'Type' in main_df.columns:
			type_dist = main_df.groupby('Type').agg({
				'Project ID': 'count',
				'Actual Trained': 'sum'
			}).reset_index()
			type_dist.columns = ['Type', 'Projects', 'Trained']
			
			fig = px.pie(type_dist, values='Projects', names='Type',
						title="Project Distribution by Type", hole=0.4)
			st.plotly_chart(fig, use_container_width=True)
			
	with col2:
		st.subheader("SPOC-wise Performance")
		if 'SPOC' in main_df.columns:
			spoc_perf = main_df.groupby('SPOC').agg({
				'Project ID': 'count',
				'Actual Trained': 'sum'
			}).reset_index()
			
			spoc_perf.columns = ['SPOC', 'Projects', 'Trained']
			spoc_perf = spoc_perf.nlargest(top_n, 'Projects')
			
			fig = px.bar(spoc_perf, x='SPOC', y='Projects',
						title=f"Top {min(top_n, len(spoc_perf))} SPOCs",
						color='Trained', color_continuous_scale='Blues',
						text='Projects')
			fig.update_layout(xaxis={'tickangle': -45})
			fig.update_traces(texttemplate='%{text}', textposition='outside')
			st.plotly_chart(fig, use_container_width=True)
			
	# Key Findings
	st.markdown("---")
	st.header("Key Findings")
	
	col1, col2, col3 = st.columns(3)
	
	with col1:
		closed_100 = len(main_df[(main_df['Fund Released %'] == 100) & (main_df['Withdrawal %'] == 100)])
		st.info(f"""
		**Financial Health**
		- Total Sanctioned: ₹{main_df['Project Cost'].sum()/1e7:.2f} Cr
		- Total Released: ₹{main_df['Total Fund Released'].sum()/1e7:.2f} Cr
		- Fully Closed (100%): {closed_100:,}
		""")
		
	with col2:
		total_trained = main_df['Actual Trained'].sum()
		total_pass = main_df['Pass'].sum()
		total_placed = main_df['Placed'].sum()
		
		st.success(f"""
		**Training Performance**
		- Total Trained: {int(total_trained):,}
		- Pass Rate: {(total_pass/total_trained*100) if total_trained > 0 else 0:.1f}%
		- Placement: {(total_placed/total_trained*100) if total_trained > 0 else 0:.1f}%
		""")
		
	with col3:
		zero_training = len(main_df[(main_df['Actual Trained'] == 0) & (main_df['In- training'] == 0)])
		recovery_proj = len(main_df[(main_df['Fund Released %'] > 100) | (main_df['Withdrawal %'] > 100)])
		
		st.warning(f"""
		**Action Required**
		- Zero Training: {zero_training:,}
		- Recovery (>100%): {recovery_proj:,}
		- Eligible for Closure: {len(main_df[(main_df['Fund Released %'] > 86) & (main_df['Fund Released %'] < 100)]):,}
		""")
		
	# Footer
	st.markdown("---")
	st.markdown("""
	<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
		<p><strong>SAMARTH - Scheme for Capacity Building in Textile Sector</strong></p>
		<p>Ministry of MSME | IT & Planning Division | Government of India</p>
		<p style='font-size: 0.9em;'>Data as on: 14 January 2026 | Powered by Streamlit & Plotly</p>
	</div>
	""", unsafe_allow_html=True)
	
if __name__ == "__main__":
	main()
	