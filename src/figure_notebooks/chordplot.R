data_path <- "/data/processed/dataframes/prepared_links.csv"

library(showtext)
library(data.table)
library(circlize)

links <- fread(data_path)
links <- as.data.table(links)

grid.col <- c(`1` = "#a40000", `2` = "#16317d", `3` = "#007e2f",
              `4` = "#ffcd12", `5` = "#b86092", `6` = "#721b3e",
              `7` = "#00b7a7", `8` = "#e35e28")

topic_names <- c('Mandatory\nVaccination',
                'Adverse\nreactions',
                'Constitutionality',
                'Ineffectiveness',
                'Long Term\nSide Effects',
                'Politics and\nConspiracy\nTheories',
                'Natural\nImmunity',
                'Others')

names(grid.col) <- topic_names

links[, c("source", "target") := .(topic_names[source], topic_names[target])]

links[source == target, value := as.integer(value/2)]

adjusted_links <- copy(links)

sector_names <- c()
new_order <- c()
counter <- 0
for (i in topic_names) {
  str_counter <- as.character(counter)
  adjusted_links[source == i & target == i & source == target,
        c("source", "target") := .(str_counter, str_counter)]
  grid.col[str_counter] <- grid.col[i]
  new_order <- append(new_order, str_counter)
  new_order <- append(new_order, i)
  
  sector_names <- append(sector_names, "")
  sector_names <- append(sector_names, i)
  counter <- counter + 1
}

showtext_auto()
pdf(file = "/figures/experimental/main_figures/user_topic_analysis_on_ChordDiagram.pdf", width=7, height=3.5)
par(mfrow = c(2, 4), cex= 0.45, mar = c(0, 2, 0, 1.1), xpd=NA)
for (i in topic_names) {
  str_counter <- as.character(counter)
  
  adjusted_links[target == i, c('source', 'target') := .(target, source)]
  
  col_mat <- ifelse(adjusted_links[, (source == i | target == i) & (source != target)],
                    grid.col[i], "#00000000")
  
  
  cd <- chordDiagram(adjusted_links, grid.col = grid.col, order = new_order,
               col = col_mat, annotationTrack = c('grid'),
               annotationTrackHeight = c(0.07, 0.05))

   circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
     xlim = get.cell.meta.data("xlim")
     ylim = get.cell.meta.data("ylim")
     sector.name = get.cell.meta.data("sector.index")
     if (sector.name %in% as.character(0:8)) {
       sector.name <- ""
     } else {
       count_newline <- length(gregexpr("\n", sector.name)[[1]])
       if (count_newline == 2) {
         ylim <- 4
       } else if (count_newline == 1) {
         ylim <- 2.8
       }
     }
     circos.text(mean(xlim), ylim, sector.name,
                 niceFacing = TRUE, cex = 0.8,
                 col=grid.col[sector.name], facing = "outside",
                 font = 2)
   }, bg.border = NA)
   
}
circos.clear()

dev.off()